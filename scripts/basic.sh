# export TRAIN_SAMPLES=3000
# export TARGET=llm
# export DATA_PATH=/work/sc126/sc126/s2598967/cachellm/cache_llm/
# export BASE_MODEL=t5-base

export RETRAIN_FREQ=1000
export BUDGET=3500
export TASK_NAME=isear
export N_INIT=100
export SEED=0
export CHECKPOINT=${SEED}_${N_INIT}

export PART=cirrus
export TAGS=CIRRUS_BATCH_INC_Test
export INCREMENTAL=yes
export BUFFER_PERCENT=0.05
export BUFFER_POLICY_PARAMETER=default
export STRATEGY=BT
# export P_STRAT=0
export P_STRAT=5
sbatch --export=ALL scripts/sub_$PART.sh