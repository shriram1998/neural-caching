#source scripts/cluster.sh
export TRAIN_SAMPLES=10000
export TARGET=llm
export DATA_PATH=/work/sc126/sc126/s2598967/cachellm/cache_llm/
export PART=cirrus
export BASE_MODEL=t5-base
export INCREMENTAL=no

# HE ENVIAT MASSES JOBS, AIXI HO HE DEIXAT!
for SEED in 2
do
    for BUFFER_POLICY_PARAMETER in default
    do
        for BUFFER_PERCENT in 1.0
        do
            for RETRAIN_FREQ in 100
            do
                for BUDGET in 3000
                do  # cr ag_news isear_llama rt-polarity_llama isear_mistral rt-polarity_mistral
                    for TASK_NAME in isear #sst2 fever_mistral openbook_mistral
                    do 
                        for STRATEGY in b1 #els deixo pel feturo!! 108 jobs funciona be
                        do
                            export BUFFER_POLICY_PARAMETER
                            export BUFFER_PERCENT
                            export TASK_NAME
                            export STRATEGY
                            export BUDGET
                            export RETRAIN_FREQ
                            export N_INIT=100
                            export TAGS=CIRRUS_BATCH_INC
                            export CHECKPOINT=${SEED}_${N_INIT}

                            if [ $STRATEGY == "b1" ]
                            then
                                export P_STRAT=0
                                sbatch --export=ALL scripts/sub_$PART.sh
                            fi
                            if [ $STRATEGY == "b2" ]
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
                                for P_STRAT in 0.9
                                do
                                    export P_STRAT
                                    export EMBED=t5
                                    sbatch --export=ALL scripts/sub_$PART.sh
                                done
                            fi
                        done
                    done  
                done
            done
        done
    done
done
