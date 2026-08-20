ds_names=(cora citeseer pubmed cora_cc citeseer_cc cora_ca acm)
noise_types=(uniform pair)

for ds_name in "${ds_names[@]}";
do
    for noise_type in "${noise_types[@]}";
    do
        if [ "$noise_type" == "uniform" ]; then
            noise_rates=(0.2 0.4 0.6)
        else
            noise_rates=(0.2 0.3 0.4)
        fi
        
        for noise_rate in "${noise_rates[@]}";
        do
            python3 main.py --dataset=${ds_name} --noise_type=${noise_type} --noise_rate=${noise_rate} --gpu 1
            wait
        done
    done
done
