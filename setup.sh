#!/usr/bin/env bash
# =============================================================================
#  NIDS Dashboard — Script de démarrage automatique
# =============================================================================
set -e

RED='\033[0;31m'; GRN='\033[0;32m'; YEL='\033[1;33m'; CYA='\033[0;36m'; NC='\033[0m'
ok()   { echo -e "${GRN}  ✓${NC} $1"; }
warn() { echo -e "${YEL}  ⚠${NC} $1"; }
info() { echo -e "${CYA}  →${NC} $1"; }
fail() { echo -e "${RED}  ✗${NC} $1"; }

echo ""
echo -e "${CYA}╔══════════════════════════════════════════════╗${NC}"
echo -e "${CYA}║     NIDS Dashboard v2.0 — Setup & Start      ║${NC}"
echo -e "${CYA}╚══════════════════════════════════════════════╝${NC}"
echo ""

# ── 1. KitNET-py ──────────────────────────────────────────────────
echo "[1/5] KitNET-py (AfterImage)"
if [ ! -d "KitNET-py" ]; then
  info "Clonage de KitNET-py..."
  git clone https://github.com/ymirsky/KitNET-py.git && ok "Cloné"
else
  ok "KitNET-py déjà présent"
fi

# ── 2. Python dependencies ────────────────────────────────────────
echo ""
echo "[2/5] Dépendances Python"
pip install -r requirements.txt -q && ok "Installées"

# ── 3. tshark ─────────────────────────────────────────────────────
echo ""
echo "[3/5] tshark / Wireshark"
if command -v tshark &>/dev/null; then
  ok "tshark $(tshark --version 2>&1 | head -1 | awk '{print $2}')"
else
  warn "tshark non trouvé"
  if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    info "Installation : sudo apt-get install -y tshark"
    read -p "  Installer maintenant ? [y/N] " yn
    [[ "$yn" == "y" || "$yn" == "Y" ]] && sudo apt-get install -y tshark
  elif [[ "$OSTYPE" == "darwin"* ]]; then
    info "Installation : brew install wireshark"
  else
    warn "Windows : installez Wireshark depuis https://www.wireshark.org"
    warn "          Ajoutez tshark.exe au PATH"
  fi
fi

# ── 4. Frontend ───────────────────────────────────────────────────
echo ""
echo "[4/5] Frontend (npm)"
if command -v npm &>/dev/null; then
  npm install -q && ok "node_modules installé"
else
  fail "npm non trouvé — installez Node.js depuis https://nodejs.org"
fi

# ── 5. Modèles ────────────────────────────────────────────────────
echo ""
echo "[5/5] Modèles ML"
mkdir -p models
MODELS=(
  "best_binary_model.pkl"
  "xgb_hierarchical_multiclass.pkl"
  "scaler_hierarchical.pkl"
  "label_encoder_hierarchical.pkl"
)
ALL_OK=true
for m in "${MODELS[@]}"; do
  if [ -f "models/$m" ]; then
    ok "models/$m"
  else
    warn "models/$m MANQUANT"
    ALL_OK=false
  fi
done

if [ "$ALL_OK" = false ]; then
  echo ""
  warn "Modèles manquants — le dashboard tournera en MODE DÉMO"
  warn "Pour le mode production, placez vos .pkl dans ./models/"
fi

# ── Résumé & démarrage ────────────────────────────────────────────
echo ""
echo -e "${CYA}══════════════════════════════════════════════${NC}"
echo ""
ok "Setup terminé"
echo ""
echo "  Démarrage :"
echo -e "  ${YEL}Terminal 1${NC} : uvicorn api.main:app --reload --host 0.0.0.0 --port 8000"
echo -e "  ${YEL}Terminal 2${NC} : npm run dev"
echo ""
echo -e "  Dashboard : ${CYA}http://localhost:5173${NC}"
echo -e "  API docs  : ${CYA}http://localhost:8000/docs${NC}"
echo ""

# Démarrage automatique ?
read -p "  Démarrer automatiquement les deux serveurs ? [y/N] " auto
if [[ "$auto" == "y" || "$auto" == "Y" ]]; then
  info "Démarrage de l'API backend..."
  uvicorn api.main:app --host 0.0.0.0 --port 8000 &
  API_PID=$!
  sleep 2

  info "Démarrage du frontend..."
  npm run dev &
  FRONTEND_PID=$!

  echo ""
  ok "Serveurs démarrés"
  echo -e "  ${CYA}http://localhost:5173${NC}"
  echo ""
  echo "  Ctrl+C pour arrêter"
  wait $API_PID $FRONTEND_PID
fi