#!/usr/bin/env python3
"""Diagnose bionty Organism state on the lamindb instance."""
import os
os.environ["TORCH_HOME"]           = "/lustre/fswork/projects/rech/xeg/uat95fg/.cache/torch"
os.environ["HF_HOME"]              = "/lustre/fswork/projects/rech/xeg/uat95fg/.hf"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/lustre/fswork/projects/rech/xeg/uat95fg/.hf/hub"

import bionty as bt
import lamindb as ln

print("=== ln.context ===")
try:
    print(f"Instance: {ln.setup.settings.instance.slug}")
except Exception as e:
    print(f"Instance err: {e}")

print()
print("=== bt.Organism local records ===")
try:
    df = bt.Organism.df()
    print(f"Total: {len(df)}")
    if len(df):
        cols = [c for c in ["name","ontology_id","scientific_name"] if c in df.columns]
        print(df[cols].head(30).to_string())
except Exception as e:
    print(f"Err: {e}")

print()
print("=== filter NCBITaxon:9606 ===")
try:
    r = bt.Organism.filter(ontology_id="NCBITaxon:9606").first()
    print("Result:", r)
except Exception as e:
    print(f"Err: {e}")

print()
print("=== bt.Gene count ===")
try:
    print(f"Total Genes in instance: {bt.Gene.df().shape[0]}")
except Exception as e:
    print(f"Err: {e}")

print()
print("=== bionty public source for Organism ===")
try:
    pub = bt.Organism.public()
    print(f"public: {pub}")
    res = pub.search("human").head(3)
    print(res.to_string())
except Exception as e:
    print(f"Public err: {e}")
