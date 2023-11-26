import NBInclude: @nbinclude
@nbinclude("PretrainDataset.ipynb")

run(`python3 Pytorch.py --outdir $name --epochs 64`)
rm(outdir, recursive = true)