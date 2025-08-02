for file in configs/DNN/*.yaml; do
    python medias.py -c "$file"
done

for file in configs/KAN1D_poly/*.yaml; do
    python medias.py -c "$file"
done

for file in configs/KAN1D_poly5/*.yaml; do
    python medias.py -c "$file"
done

for file in configs/KAN2D_gen/*.yaml; do
    python medias.py -c "$file"
done

for file in configs/KAN2D_poly/*.yaml; do
    python medias.py -c "$file"
done


