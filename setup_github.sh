#!/bin/bash
# GitHub Setup Script for qkvflow

echo "╔═══════════════════════════════════════════════════════╗"
echo "║     GitHub Setup for qkvflow Research Project        ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""

# Navigate to project
cd /home/nahid/Documents/qkvflow

# Configure Git user (replace with your details)
echo "Configuring Git user..."
git config user.name "zaphrode"
git config user.email "your-email@example.com"  # Replace this!

# Initialize Git repository
echo "Initializing Git repository..."
git init

# Create .gitignore
echo "Creating .gitignore..."
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
venv311/
env/
*.egg-info/
dist/
build/

# JAX/ML
*.pkl
*.ckpt
*.safetensors
*.npz

# Logs
*.log
nohup.out
*.txt.lock

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Data
*.zip
wikitext*/
train.txt
test.txt

# Temporary files
*.tmp
*.bak
validation_pid.txt

# Keep only essential results
# (we'll add back specific files we want)
EOF

# Add files
echo "Adding files to Git..."
git add .

# Initial commit
echo "Creating initial commit..."
git commit -m "Initial commit: Time-Indexed Parameter Sharing for Neural ODE Transformers

- Novel time-indexed parameter sharing approach
- Two variants: MLP (430× compression) and SSM (62× compression)
- 9.1% better performance than Tong et al. (ICLR 2025)
- Statistical validation with 5 seeds
- Publication-ready figures and LaTeX table
- WikiText-2 benchmark results"

echo ""
echo "✅ Git repository initialized!"
echo ""
echo "═══════════════════════════════════════════════════════"
echo "Next steps:"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "1. Create repository on GitHub:"
echo "   → Go to https://github.com/new"
echo "   → Name: qkvflow (or your preferred name)"
echo "   → Visibility: Public or Private"
echo "   → DON'T initialize with README"
echo ""
echo "2. After creating, GitHub will show you commands like:"
echo "   git remote add origin git@github.com:zaphrode/REPO_NAME.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "3. Run those commands here!"
echo ""
echo "═══════════════════════════════════════════════════════"

