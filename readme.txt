# Initialize virtual env
python3 -m venv .venv

# Launch virtual env
source .venv/bin/activate

# Install requirements
python -m pip install -r requirements.txt

# Run main script
python main.py