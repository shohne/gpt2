#!/bin/bash

if [ $# -eq 0 ]; then
    MAX_SHARDS=1001
else
    MAX_SHARDS=$1
fi

if [ $MAX_SHARDS -gt 1001 ]; then
    MAX_SHARDS=1001
fi

TRAIN_BASE_URL="https://huggingface.co/datasets/karpathy/fineweb-edu-100B-gpt2-token-shards/resolve/main/edu_fineweb_train_"
VAL_URL="https://huggingface.co/datasets/karpathy/fineweb-edu-100B-gpt2-token-shards/resolve/main/edu_fineweb_val_000000.bin"

SAVE_DIR="./dev/data/edu_fineweb_100B"

mkdir -p $SAVE_DIR

download() {
    local FILE_URL=$1
    local FILE_NAME=$(basename $FILE_URL | cut -d'?' -f1)
    local FILE_PATH="${SAVE_DIR}/${FILE_NAME}"
    curl -s -L -o "$FILE_PATH" "$FILE_URL"
    echo "Downloaded $FILE_URL to $SAVE_DIR"
}

run_in_parallel() {
    local max_jobs=$1
    shift
    local commands=("$@")
    local job_count=0

    for cmd in "${commands[@]}"; do
        eval "$cmd" &
        ((job_count++))
        if (( job_count >= max_jobs )); then
            wait -n
            ((job_count--))
        fi
    done
    wait
}

export -f download

download "$VAL_URL" &

train_commands=()
for i in $(seq -f "%06g" 1 $MAX_SHARDS); do
    FILE_URL="${TRAIN_BASE_URL}${i}.bin?download=true"
    train_commands+=("download \"$FILE_URL\"")
done

run_in_parallel 40 "${train_commands[@]}"
echo "The val shard first $MAX_SHARDS train shards of FineWebEdu100B files downloaded in $SAVE_DIR"

