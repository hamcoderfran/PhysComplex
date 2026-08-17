Quick Start:

cd physcomplex_filter
pip install -e .
physrna init
physrna doctor

# Score AF3 Server zips
python -m physrna_filter.validation.screen_af3 /path/to/af3_zips --fast -o screen.csv
physrna rank /path/to/af3_zips --rbp HUD --fast --no-finetune

# Single structure
python -m physrna_filter.pipeline physrna_filter/data/structures/1urn.pdb

# PhysGT eval
python -m physrna_filter.validation.eval_gt --max-entries 50
