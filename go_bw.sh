#!/bin/bash

RCs=(1 2)
columns=(`seq 0 15`)
#RCs=(2)
#columns=(5)

OUTFILE=$1
BYROWCOL=false

# Superconducting, run 13
# After remounting MCE with explicit grounding on cryostat and MCE
# sides of the cable, and 3x turns now (was 2x) for each cable from
# cryostat to MCE around the ferrite torroid stack.
OUTFILE=$1
BYROWCOL=false

MCE_DATADIR=/mnt/mce

#http://stackoverflow.com/questions/1527049/bash-join-elements-of-an-array
function join { local IFS="$1"; shift; echo "$*"; }

if [ "$BYROWCOL" = true ]; then
    python write_raw_to_tods.py ${OUTFILE} -p ${MCE_DATADIR} -c `join , ${columns[@]}` -r `join , ${rows[@]}`
else
    python write_raw_to_tods.py ${OUTFILE} -p ${MCE_DATADIR} -c `join , ${columns[@]}`
fi

# plotting directives
gain=189
nV_per_ADU=707.5
infile=${OUTFILE%.*}.in
echo -e "title=`date "+%m/%d/%y%n"` ${OUTFILE}" > ${infile}
echo -e "ylabel=ASD (nV/rtHz w/ G=${gain}, nV/ADU=${nV_per_ADU})" > ${infile}
echo -e "xmin=1e3" >> ${infile}
echo -e "ymin=6e-2" >> ${infile}
echo -e "ymax=6e2" >> ${infile}
echo -e "# islog='xy'" >> ${infile}
echo -e "filter_size=50" >> ${infile}
echo -e "color_order=-1" >> ${infile}
echo -e "figfilename=${OUTFILE%.*}.png" >> ${infile}
echo -e "daqoutfile(s)=${OUTFILE}" >> ${infile}

# which RC?
function RC_by_col {
    local COL=$1
    local RC=
    if [ \( $COL -lt 8 \) -a \( $COL -ge 0 \) ]; then 
	RC=1; 
    elif [ \( $COL -lt 16 \) -a \( $COL -ge 8 \) ]; then
	RC=2; 
    elif [ \( $COL -lt 24 \) -a \( $COL -ge 16 \) ]; then
	RC=3; 
    elif [ \( $COL -lt 32 \) -a \( $COL -ge 24 \) ]; then
	RC=4; 
    else
	print "! Can't figure out what RC this COL=$COL is on.  Abort!"
	exit
    fi
    echo ${RC}
}
print "Testing 1"
# list contains?
# http://stackoverflow.com/questions/3685970/check-if-an-array-contains-a-value
function contains() {
    local n=$#
    local value=${!n}
    for ((i=1;i < $#;i++)) {
        if [ "${!i}" == "${value}" ]; then
            echo "y"
            return 0
        fi
    }
    echo "n"
    return 1
}

print "Testing 2"
if [ "$BYROWCOL" = true ]; then

    OUTFILEPATH=`python path_to_outfile.py ${OUTFILE}`
    # this is insanity - assumes first row is header with column names; looks for columns named "row" and "col"
    while read row col; do
	if [ \( $(contains "${columns[@]}" "$col") == "y" \) -a \( $(contains "${rows[@]}" "$row") == "y" \) ]; then
	    rc=$(RC_by_col ${col})
	    echo ${rc} ${row} ${col}
	    suffix=r${row}c${col}
	    python compute_fft_from_raw.py ${OUTFILE} -b -r ${row} -c ${col} -s ${suffix} --rc ${rc}
	    echo -e "`echo ${OUTFILE%%.*}`_rc${rc}_c${col}_${suffix}.meanfft.bz2  `bc <<< "scale=2; ${nV_per_ADU}"` ${suffix}" >> ${infile}	
	fi
    done < <(awk -vcol1="row" -vcol2="col" 'NR==1{for(i=1;i<=NF;i++){if($i==col1)c1=i; if ($i==col2)c2=i;}} NR>1{print $c1 " " $c2}'  ${OUTFILEPATH} | uniq)

else # assume one dataset per col
    for rc in ${RCs[@]}; do
	for c in ${columns[@]}; do 
	    col_for_file=$((${c}+$(($((${rc}-1))*8))))
	    python compute_fft_from_raw.py ${OUTFILE} -c ${col_for_file} --rc ${rc} -p ${MCE_DATADIR}
	    OUTFILE_SANS_SUFFIX=`echo ${OUTFILE%%.*}`
	    if [ -f "./output/${OUTFILE_SANS_SUFFIX}/${OUTFILE_SANS_SUFFIX}_rc${rc}_c${col_for_file}.meanfft.bz2" ]; then
		echo -e "${OUTFILE_SANS_SUFFIX}_rc${rc}_c${col_for_file}.meanfft.bz2  `bc <<< "scale=2; ${nV_per_ADU}"` RC${rc}_c${c}" >> ${infile}
	    fi
	done
    done
fi
