"""
ARV-DS03 — Interface web de detection de pneumonie
==================================================
Deux modes :
  1. MedGemma (via Ollama) + YOLO — analyse EN DIRECT d'une radio DICOM.
     Cascade : MedGemma decide (normal / suspected_opacity), puis YOLO confirme/localise.
  2. MAIRA-2 + GPT-OSS — PRE-ANALYSE (lecture du CSV genere sur Kaggle).

Lancement :  streamlit run app/app.py

Prerequis (voir README.md) :
  - Ollama installe et lance, avec le modele MedGemma tire (ollama pull ...).
  - Le fichier de poids YOLO best.pt.
  - Le CSV data/maira2_interface.csv (pour le mode 2).
"""
import os
import io
import json
import base64
import csv
from datetime import datetime
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

st.set_page_config(page_title="ARV-DS03 — Detection pneumonie", layout="wide")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(APP_DIR)

# ============================================================
# CONFIGURATION — barre laterale (tes 3 ressources)
# ============================================================
st.sidebar.header("Configuration")
YOLO_WEIGHTS = st.sidebar.text_input("1. Poids YOLO (best.pt)", "best.pt")
OLLAMA_MODEL = st.sidebar.text_input("2. Modele MedGemma (nom Ollama)", "dcarrascosa/medgemma-1.5-4b-it:Q4_K_M")
MAIRA_CSV    = st.sidebar.text_input("3. CSV MAIRA pre-analyse", os.path.join(REPO_ROOT, "data", "maira2_interface.csv"))
OLLAMA_URL   = st.sidebar.text_input("URL Ollama", "http://localhost:11434")
USE_MEDGEMMA = st.sidebar.checkbox(
    "Activer MedGemma (description)", value=True,
    help="YOLO detecte toujours ; MedGemma redige la description si active.",
)
# YOLO decide la classe. Bande d'incertitude : une detection dont la confiance
# est entre ces deux seuils est classee 'uncertain' (detection peu sure).
YOLO_UNCERTAIN_LOW = st.sidebar.slider(
    "Seuil bas YOLO (sous = ignore)", 0.0, 1.0, 0.25, 0.05,
    help="Detections sous ce seuil ignorees.",
)
YOLO_UNCERTAIN_HIGH = st.sidebar.slider(
    "Seuil haut YOLO (au-dessus = opacite confirmee)", 0.0, 1.0, 0.45, 0.05,
    help="Entre les deux seuils : incertain. Au-dessus : pneumonie confirmee.",
)

# Parametres YOLO (repris de ton notebook)
IMGSZ, CONF_THRES, SHRINK, NMS_IOU = 640, 0.10, 0.85, 0.4  # conf bas : la bande incertain est geree en aval
CLAHE_CLIP, CLAHE_GRID = 2.0, (8, 8)

# --- Barre laterale : telechargement du journal CSV ---
_log_path = os.path.join(APP_DIR, "journal_analyses.csv")
if os.path.isfile(_log_path):
    with open(_log_path, "rb") as _f:
        st.sidebar.download_button(
            "📥 Telecharger le journal (CSV)", _f, file_name="journal_analyses.csv",
            mime="text/csv",
        )
    st.sidebar.caption(f"Journal : {sum(1 for _ in open(_log_path, encoding='utf-8-sig'))-1} analyses enregistrees.")

COLORS = {"normal": "#2e9e4f", "suspected_opacity": "#d1342f", "uncertain": "#e6a817"}
LABEL_FR = {"normal": "Normal", "suspected_opacity": "Pneumonie suspectee", "uncertain": "Incertain"}

# ============================================================
# Journalisation des analyses (CSV dans le repertoire de l'app)
# ============================================================
LOG_CSV = os.path.join(APP_DIR, "journal_analyses.csv")
LOG_COLUMNS = [
    "horodatage", "fichier", "classe_finale", "nb_boites",
    "confiances_yolo", "boites_yolo",
    "medgemma_actif", "image_quality", "confidence",
    "justification", "visual_evidence", "warning",
]

def log_analysis(filename, final_class, pred_boxes, mg):
    """Ajoute une ligne au CSV de journalisation. Cree le fichier + l'en-tete si besoin."""
    confs = ";".join(f"{b[4]:.3f}" for b in pred_boxes) if pred_boxes else ""
    boxes = ";".join(
        f"({int(b[0])},{int(b[1])},{int(b[2])},{int(b[3])})" for b in pred_boxes
    ) if pred_boxes else ""

    if mg is not None:
        ve = mg.get("visual_evidence", [])
        ve_str = " | ".join(str(x) for x in ve) if isinstance(ve, list) else str(ve)
        row = {
            "medgemma_actif": "oui",
            "image_quality": mg.get("image_quality", ""),
            "confidence": mg.get("confidence", ""),
            "justification": str(mg.get("justification", "")).replace("\n", " "),
            "visual_evidence": ve_str,
            "warning": mg.get("warning", ""),
        }
    else:
        row = {k: "" for k in ["image_quality", "confidence", "justification",
                               "visual_evidence", "warning"]}
        row["medgemma_actif"] = "non"

    row.update({
        "horodatage": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "fichier": filename,
        "classe_finale": final_class,
        "nb_boites": len(pred_boxes),
        "confiances_yolo": confs,
        "boites_yolo": boxes,
    })

    file_exists = os.path.isfile(LOG_CSV)
    try:
        with open(LOG_CSV, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        return True, LOG_CSV
    except Exception as e:
        return False, str(e)

# ============================================================
# YOLO (charge une seule fois, mis en cache)
# ============================================================
@st.cache_resource(show_spinner="Chargement de YOLO...")
def load_yolo(weights):
    from ultralytics import YOLO
    return YOLO(weights)

# ============================================================
# Traitement image
# ============================================================
def dicom_to_rgb(dcm_bytes, size=IMGSZ):
    import cv2, pydicom
    dcm = pydicom.dcmread(io.BytesIO(dcm_bytes))
    img = dcm.pixel_array.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = (img * 255.0).astype(np.uint8)
    if img.shape[0] != size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    return cv2.cvtColor(clahe.apply(img), cv2.COLOR_GRAY2RGB)

def shrink_box(x1, y1, x2, y2, s):
    cx, cy = (x1+x2)/2, (y1+y2)/2
    w, h = (x2-x1)*s, (y2-y1)*s
    return cx-w/2, cy-h/2, cx+w/2, cy+h/2

def predict_boxes(yolo, rgb):
    res = yolo.predict(rgb, conf=CONF_THRES, iou=NMS_IOU, imgsz=IMGSZ, verbose=False)[0]
    out = []
    for b in res.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        x1, y1, x2, y2 = shrink_box(x1, y1, x2, y2, SHRINK)
        out.append((x1, y1, x2, y2, float(b.conf[0])))
    return out

def draw_boxes(rgb, pred_boxes, gt_boxes=None):
    img = Image.fromarray(rgb).convert("RGB")
    d = ImageDraw.Draw(img)
    if gt_boxes:
        for (x1, y1, x2, y2) in gt_boxes:
            d.rectangle([x1, y1, x2, y2], outline=(60, 200, 90), width=3)
    for b in pred_boxes:
        x1, y1, x2, y2 = b[:4]
        d.rectangle([x1, y1, x2, y2], outline=(230, 60, 50), width=3)
        if len(b) >= 5:
            d.text((x1, max(0, y1-12)), f"{b[4]:.2f}", fill=(230, 60, 50))
    return img

# ============================================================
# MedGemma via Ollama (serveur local, requete HTTP)
# ============================================================
# MedGemma est ici REDACTEUR (pas decideur) : YOLO detecte, MedGemma decrit.
# On lui fournit l'image ET le verdict de YOLO, il produit le JSON descriptif coherent.
# Prompt complet : prompts/medgemma_report_writer_prompt.txt
def build_medgemma_prompt(yolo_class, yolo_boxes_desc):
    """Construit le prompt en injectant le resultat de YOLO."""
    return f"""You are simulating the report-writing style of an experienced radiologist reviewing a frontal chest X-ray (PA/AP view). This is for educational purposes only.

CONTEXT (internal use only — do NOT reference this context, its source, or any model/algorithm in your output):
- Working verdict to be consistent with: {yolo_class}
- Anatomical zones of interest: {yolo_boxes_desc}

Your job is to WRITE THE RADIOLOGY REPORT ITSELF, as if you are the one interpreting the image directly.
You must NEVER write phrases like "the detection model", "YOLO", "the algorithm identified", "a region flagged as", "the AI detected", etc.
Instead, describe findings the way a radiologist dictates a report: direct, anatomical, declarative statements about what is seen in the image.

Bad style (forbidden — sounds like commentary on another model's output):
"The detection model identified regions of increased density consistent with opacity. The lower left lung field shows a distinct opacity."

Good style (required — reads like an actual radiology report):
"There is a focal area of increased opacity in the left lower lung zone, with ill-defined margins, suggestive of an airspace consolidation. The right lung remains clear. The cardiac silhouette is within normal limits (cardiothoracic ratio < 0.5). Costophrenic angles are sharp bilaterally. No pneumothorax or pleural effusion identified. Mediastinal contours are unremarkable."

SYSTEMATIC CHECKLIST — your "justification" and "visual_evidence" must, where relevant, address:
- Lung fields (upper/mid/lower, left/right): clear vs opacity/consolidation/infiltrate, distribution (focal, diffuse, bilateral)
- Cardiac silhouette: size and cardiothoracic ratio (within normal limits vs cardiomegaly)
- Mediastinum: width, contours (widened vs normal)
- Costophrenic angles: sharp vs blunted (effusion)
- Pleura: pneumothorax, effusion, thickening
- Bones/soft tissue: fractures, subcutaneous emphysema (only mention if visually plausible, otherwise state unremarkable)
- Diaphragm: position, contour

Output strictly valid JSON matching this schema:
{{
  "image_quality": "good | limited | poor",
  "predicted_class": "{yolo_class}",
  "confidence": <float between 0.0 and 1.0>,
  "visual_evidence": ["short, factual radiological observation 1", "observation 2", "..."],
  "justification": "A cohesive, well-written radiology-report-style paragraph (3-5 sentences) describing all clinically relevant zones, written as direct findings — never referencing a detection process.",
  "limitations": ["limitation 1", "..."],
  "warning": "Prototype pedagogique. Non destine au diagnostic medical."
}}

Rules:
1. "predicted_class" MUST equal: {yolo_class}. Never contradict it, but do not name it as a "verdict" in the readable text fields.
2. If {yolo_class} is "normal": describe clear lung fields, normal cardiac silhouette, sharp costophrenic angles, no acute abnormality — a full normal report, not just "nothing found".
3. If {yolo_class} is "suspected_opacity": describe the opacity's location, shape, margins, and density using the anatomical zones provided, plus confirm normal findings elsewhere (heart, pleura, bones) to give a complete picture.
4. If {yolo_class} is "uncertain": explain concretely which findings are ambiguous (e.g., low image quality, overlapping structures, borderline density) rather than vaguely stating uncertainty.
5. Never use meta-language about models, detections, algorithms, or confidence pipelines anywhere in "visual_evidence" or "justification".
6. Output ONLY valid JSON, nothing else. No comments, no markdown, no thought process.
"""

def medgemma_describe_ollama(rgb, yolo_class, yolo_boxes_desc, model_name, url):
    """MedGemma REDIGE le JSON descriptif a partir du verdict de YOLO.
    Retourne (dict_json, ok). MedGemma ne decide pas la classe : il decrit."""
    import requests
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    prompt = build_medgemma_prompt(yolo_class, yolo_boxes_desc)
    try:
        resp = requests.post(
            f"{url}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt + "\n\nAnalyze the following image.",
                "images": [img_b64],
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=180,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
    except Exception as e:
        return {"error": str(e)}, False

    try:
        s, e = raw.find("{"), raw.rfind("}") + 1
        if s == -1 or e <= s:
            raise ValueError("Aucun JSON detecte")
        pred = json.loads(raw[s:e])
    except Exception:
        pred = {
            "image_quality": "limited",
            "predicted_class": yolo_class,
            "confidence": 0.0,
            "visual_evidence": [],
            "justification": "Description non parsable ; verdict fourni par YOLO.",
            "limitations": ["parsing failed"],
            "warning": "Prototype pedagogique. Non destine au diagnostic medical.",
            "_raw": raw,
        }
    # La classe reste celle de YOLO, quoi que dise le parsing
    pred["predicted_class"] = yolo_class
    return pred, True

def medgemma_boxes_to_pixels(pred, size):
    """Convertit les box_2d [y1,x1,y2,x2] (echelle 0-1024) de MedGemma en pixels (x1,y1,x2,y2)."""
    out = []
    for op in pred.get("opacities", []) or []:
        box = op.get("box_2d")
        if box and isinstance(box, (list, tuple)) and len(box) == 4 and all(v is not None for v in box):
            y1, x1, y2, x2 = [float(c) / 1024.0 for c in box]
            out.append((round(x1*size), round(y1*size), round(x2*size), round(y2*size)))
    return out

# ============================================================
# INTERFACE
# ============================================================
st.title("🫁 ARV-DS03 — Assistant de detection de pneumonie")
st.caption("Prototype pedagogique — non destine au diagnostic medical.")

mode = st.radio(
    "Choisir le modele d'analyse :",
    ["MedGemma + YOLO (analyse en direct)", "MAIRA-2 + GPT-OSS (banque pre-analysee)"],
    horizontal=True,
)

# ---------------------------------------------------------------
# MODE 1 : LIVE — MedGemma (Ollama) + YOLO
# ---------------------------------------------------------------
if mode.startswith("MedGemma"):
    st.info("Depose une radio DICOM (.dcm). L'analyse se fait en direct — l'image seule suffit.")
    up = st.file_uploader("Radiographie thoracique (DICOM)", type=["dcm"])

    if up is not None:
        rgb = dicom_to_rgb(up.read())

        with st.spinner("Analyse en cours..."):
            yolo = load_yolo(YOLO_WEIGHTS)

            # ---- ETAPE 1 : YOLO detecte et DECIDE la classe ----
            raw_boxes = predict_boxes(yolo, rgb)  # (x1,y1,x2,y2,conf)

            # Bande d'incertitude sur la confiance YOLO
            strong = [b for b in raw_boxes if b[4] >= YOLO_UNCERTAIN_HIGH]
            weak   = [b for b in raw_boxes if YOLO_UNCERTAIN_LOW <= b[4] < YOLO_UNCERTAIN_HIGH]

            if strong:
                final = "suspected_opacity"
                pred_boxes = strong
            elif weak:
                final = "uncertain"          # detection existante mais peu sure
                pred_boxes = weak
            else:
                final = "normal"             # rien de detecte -> normal
                pred_boxes = []

            # Description textuelle des boites pour MedGemma
            if pred_boxes:
                boxes_desc = "; ".join(
                    f"box(x1={int(b[0])},y1={int(b[1])},x2={int(b[2])},y2={int(b[3])},conf={b[4]:.2f})"
                    for b in pred_boxes
                )
            else:
                boxes_desc = "none (no opacity detected)"

            # ---- ETAPE 2 : MedGemma REDIGE la description (si active) ----
            mg = None
            if USE_MEDGEMMA:
                mg, ok = medgemma_describe_ollama(
                    rgb, final, boxes_desc, OLLAMA_MODEL, OLLAMA_URL
                )
                if not ok:
                    st.warning(f"MedGemma (Ollama) injoignable : {mg.get('error','?')}. "
                               f"Le verdict YOLO est conserve, sans description MedGemma.")
                    mg = None

            # ---- JOURNALISATION : sauvegarde de l'analyse dans le CSV ----
            log_ok, log_info = log_analysis(up.name, final, pred_boxes, mg)

        # Confirmation de journalisation
        if log_ok:
            st.caption(f"📝 Analyse enregistree dans : {log_info}")
        else:
            st.caption(f"⚠️ Journalisation impossible : {log_info}")

        # ---------------- Affichage ----------------
        c1, c2 = st.columns([1, 1])
        with c1:
            st.image(draw_boxes(rgb, pred_boxes),
                     caption="Rouge = zone detectee par YOLO", use_container_width=True)
        with c2:
            st.markdown(f"### Resultat : "
                        f"<span style='color:{COLORS[final]}'>{LABEL_FR[final]}</span>",
                        unsafe_allow_html=True)

            # Etape YOLO (decideur)
            st.markdown("#### Detection YOLO")
            if pred_boxes:
                confs = ", ".join(f"{b[4]:.0%}" for b in pred_boxes)
                st.write(f"**Boites :** {len(pred_boxes)}  ·  **Confiance :** {confs}")
            else:
                st.write("Aucune opacite detectee.")
            # Etape MedGemma (redacteur)
            if mg is not None:
                st.markdown("#### Description MedGemma")
                conf = mg.get("confidence", 0.0)
                st.write(f"**Qualite image :** {mg.get('image_quality', 'n/a')}")
                if mg.get("justification"):
                    st.write(f"**Justification :** {mg['justification']}")
                ve = mg.get("visual_evidence", [])
                if ve:
                    st.write("**Observations visuelles :**")
                    for obs in ve:
                        st.write(f"- {obs}")
                with st.expander("JSON complet MedGemma"):
                    st.json(mg)

            # Verdict
            if final == "uncertain":
                st.warning("Detection presente mais peu sure → incertain.")
            elif final == "suspected_opacity":
                st.error("Pneumonie suspectee et localisee par YOLO.")
            else:
                st.success("Aucune opacite : classe normale.")

        st.caption("⚠️ Prototype pedagogique. Non destine au diagnostic medical.")

# ---------------------------------------------------------------
# MODE 2 : PRE-ANALYSE — MAIRA-2 + GPT-OSS
# ---------------------------------------------------------------
else:
    st.info("Ce mode lit les resultats pre-calcules (MAIRA-2 + GPT-OSS). "
            "L'image doit faire partie de la banque pre-analysee.")
    import pandas as pd

    @st.cache_data
    def load_bank(path):
        return pd.read_csv(path)

    try:
        bank = load_bank(MAIRA_CSV)
    except Exception as e:
        st.error(f"Impossible de lire le CSV MAIRA ({MAIRA_CSV}) : {e}")
        st.stop()

    st.caption(f"Banque : {len(bank)} images pre-analysees.")
    up = st.file_uploader("Radiographie DICOM (pour l'affichage)", type=["dcm"])
    manual_id = st.text_input("...ou saisir directement un patientId")

    pid = None
    if up is not None:
        pid = os.path.splitext(up.name)[0]
    elif manual_id.strip():
        pid = manual_id.strip()

    if pid:
        row = bank[bank["patientId"] == pid]
        if row.empty:
            st.warning(f"L'image '{pid}' n'est pas dans la banque pre-analysee "
                       f"({len(bank)} images disponibles). Ce mode ne fonctionne que "
                       f"sur les images deja traitees sur Kaggle.")
        else:
            r = row.iloc[0]
            final = r["classe_finale"]
            pred_boxes = json.loads(r["boites_predites"]) if str(r["boites_predites"]).strip() else []
            gt_raw = json.loads(r["boites_reelles"]) if str(r["boites_reelles"]).strip() else []
            ref = int(r.get("ref_size", 1024))

            if up is not None:
                up.seek(0)
                rgb = dicom_to_rgb(up.read(), size=ref)
            else:
                rgb = np.full((ref, ref, 3), 30, dtype=np.uint8)

            pb = [(b["x1"], b["y1"], b["x2"], b["y2"]) for b in pred_boxes]
            gb = [(b["x1"], b["y1"], b["x2"], b["y2"]) for b in gt_raw]

            c1, c2 = st.columns([1, 1])
            with c1:
                st.image(draw_boxes(rgb, pb, gb),
                         caption="Rouge = MAIRA-2 · Vert = verite terrain RSNA",
                         use_container_width=True)
            with c2:
                st.markdown(f"### Resultat : "
                            f"<span style='color:{COLORS.get(final,'#888')}'>"
                            f"{LABEL_FR.get(final, final)}</span>", unsafe_allow_html=True)
                st.write(f"**Classe (regles) :** {LABEL_FR.get(r['classe_regles'], r['classe_regles'])}")
                if str(r.get("classe_LLM", "")).strip():
                    st.write(f"**Classe (GPT-OSS) :** {LABEL_FR.get(r['classe_LLM'], r['classe_LLM'])}")
                st.write(f"**Boites MAIRA-2 :** {len(pb)}")
                st.write(f"**Rapport MAIRA-2 :**")
                st.write(r["rapport_maira"])
            st.caption("⚠️ Prototype pedagogique. Non destine au diagnostic medical.")
