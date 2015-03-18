#!/usr/bin/env sh

run () {

    for model in $models; do
        for core in $cores; do
            for workers in $skel_workers; do
                for ops in $operators; do
                    for run in `seq 1 $run_repeat`; do
                        mkdir -p $output_root/$ops/$model/$core/w$workers

                        echo "running $model in $rtime mlsecs with $workers skel workers on $core cores with $ops operators.."
                        logfile="emas_$rtime"`date +"-%s"`".log"

                        output_file=$output_root/$ops/$model/$core/w$workers/$logfile
                        echo $output_file
                        erl +S 4:$core -pa ebin -pa deps/*/ebin \
                            -eval "emas:start($rtime,[{model,$model},{skel_workers,$workers},{genetic_ops,$ops},{problem_size,$problem_size}])." \
                            -run init stop -noshell
                        #> $output_file
                    done
                done
            done
        done
    done
}


output_dir="output"
# rtime=120000
rtime=2000

# cores="1 2 4"
cores="4"
# run_repeat=3
run_repeat=1
skel_workers=4
models="mas_skel mas_sequential mas_concurrent mas_hybrid"
operators="labs_ops labs_bin_ops"
problem_size=22

output_root=$output_dir/tests

run
