#source scripts/cluster.sh
export TRAIN_SAMPLES=10000
export TARGET=llm
export DATA_PATH=/work/sc126/sc126/s2598967/cachellm/cache_llm/
export PART=cirrus
export BASE_MODEL=t5-base

for SEED in 0 1 2
do
    for RETRAIN_FREQ in 100 # AIXO HO HAURE DE CANVIAR!!
    do
        for BUDGET in 1000
        do
            for TASK_NAME in openbook fever #fever_mistral openbook_mistral fever_llama openbook_llama #cr ag_news isear_llama rt-polarity_llama fever_llama openbook_llama isear_mistral rt-polarity_mistral fever_mistral openbook_mistral
            do 
                for STRATEGY in b1
                do
                    export SEED
                    export TASK_NAME
                    export SAVE_CHECKPOINT=yes
                    export STRATEGY
                    export BUDGET
                    export RETRAIN_FREQ
                    export TAGS=make_checkpoints_cirrus
                    export N_INIT=100 # HAURE DANAR AMB CUIDAO!!!

                    if [ $STRATEGY == "b1" ]
                    then
                        export P_STRAT=0
                        sbatch --export=ALL scripts/sub_$PART.sh
                    fi
                    if [ $STRATEGY == "BT" ]
                    then 
                        for P_STRAT in 5
                        do
                            export P_STRAT
                            sbatch --export=ALL scripts/sub_$PART.sh
                        done
                    fi
                    if [ $STRATEGY == "MV" ]
                    then
                        export P_STRAT=3
                        sbatch --export=ALL scripts/sub_$PART.sh
                    fi
                    if [ $STRATEGY == "EN" ]
                    then
                        for P_STRAT in 0.5
                        do
                            export P_STRAT
                            sbatch --export=ALL scripts/sub_$PART.sh
                        done
                    fi 
                    if [ $STRATEGY == "CS" ]
                    then
                        for P_STRAT in 0.7
                        do
                            export P_STRAT
                            export EMBED=mpnet
                            sbatch --export=ALL scripts/sub_$PART.sh
                        done
                    fi
                done
            done  
        done
    done
done