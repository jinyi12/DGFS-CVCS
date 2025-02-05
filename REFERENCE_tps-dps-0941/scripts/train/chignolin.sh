python src/train.py \
    --molecule chignolin \
    --start_state unfolded \
    --end_state folded \
    --num_steps 5000 \
    --sigma 0.5 \
    --num_rollouts 100 \
    --batch_size 4 \
    --buffer_size 200 \
    --bias scale
