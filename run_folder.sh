for file in configs/sampway_lhs_const/*.yaml; do
    python main_pl.py -c "$file"
    python medias.py -c "$file"
done
