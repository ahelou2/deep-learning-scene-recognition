train_files=`ls train*`
test_files=`ls test*`
for c in 100 200 500 1000 2000 5000
do
    for f in ${train_files}
        do
            ./svm_multiclass_learn -c ${c} -f 10 ${f} ${f}-${c}-model
        done

    for f in ${test_files}
        do
            ./svm_multiclass_classify ${f} ${f}-${c}-model ${f}-${c}-result > ${f}-${c}-out
        done
done
