python sin_freq.py --do_summary --seed 3407 \
  --width 128 --layers 3 --steps 8000 --batch 1024 --lr 1e-3 \
  --num_grids 16 \
  --freq 192 --shrink 6e-6 --l1_spec 1e-6 \
  --device cuda
