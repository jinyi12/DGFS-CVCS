python src/sample.py \
    --molecule bba \
    --start_state unfolded \
    --end_state folded \
    --num_steps 5000 \
    --sigma 0.5 \
    --bias scale \
    --model_path models/bba/scale.pt \
    --temperature 400
