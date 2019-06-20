DATADIR=data
PROG=program/SequenceTagger.py
OPTFUNC=adam
ITER_EPOCH=1
MAX_ITER=3
NUM_RECORD=3
if [ ! $1 ]; then
	echo "Usage: $0 <model (rnn | irnn | gru | igru | lstm | ilstm | imemn2n-tied | memn2n-tied)> <backend (theano | tensorflow; optional)> <GPU id; optional>"
else
	if [ ! $2 ]; then
		echo "Default is CPU for theano or automatic setting for tensorflow depending on what the backend is using."
		GPUSET=''
	elif [ $2 == 'theano' ]
	then
		GPUSET="THEANO_FLAGS=device=gpu$3,floatX=float32"
	fi
	MDL=$1
	TRAIN=$DATADIR/sample.iob
	TEST=$DATADIR/sample.iob
	TLEN=48
	VDIM=100
	for DROPOUT in 0.50
	do
		for HDIM in 50
		do
			if [ ! -d sample ]; then
				mkdir -p sample
			fi
			RES_PATH=sample
			MDL_PATH=sample
			CMD="$GPUSET python $PROG --train $TRAIN --test $TEST --sgdtype $OPTFUNC --arch $MDL --iter_per_epoch $ITER_EPOCH --out $RES_PATH -m $MAX_ITER --mdl_path $MDL_PATH --record_epoch $NUM_RECORD --dropout True --dropout_ratio $DROPOUT --hidden_size $HDIM --time_length $TLEN --embedding_size $VDIM --input_type embedding"
			echo $CMD
		done
	done
fi
