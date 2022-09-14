from process_davis_kiba import *

hp = HyperParameter()
XD, XT, affinities = read_davis_kiba(hp)
label_row_inds, label_col_inds = np.where(np.isnan(affinities) == False)
count = np.zeros(6)
count1 = np.zeros(11)
for ind in range(len(label_row_inds)):
    Y_value = pd.Series(affinities[label_row_inds[ind]][label_col_inds[ind]])
    if hp.sub == 'davis':
        if Y_value.between(5, 6).bool():
            count[0] += 1
        elif Y_value.between(6, 7).bool():
            count[1] += 1
        elif Y_value.between(7, 8).bool():
            count[2] += 1
        elif Y_value.between(8, 9).bool():
            count[3] += 1
        elif Y_value.between(9, 10).bool():
            count[4] += 1
        elif Y_value.between(10, 11).bool():
            count[5] += 1
    else:
        if Y_value.between(8, 9).bool():
            count1[0] += 1
        elif Y_value.between(9, 10).bool():
            count1[1] += 1
        elif Y_value.between(10, 11).bool():
            count1[2] += 1
        elif Y_value.between(11, 12).bool():
            count1[3] += 1
        elif Y_value.between(12, 13).bool():
            count1[4] += 1
        elif Y_value.between(13, 14).bool():
            count1[5] += 1
        elif Y_value.between(14, 15).bool():
            count1[6] += 1
        elif Y_value.between(15, 16).bool():
            count1[7] += 1
        elif Y_value.between(16, 17).bool():
            count1[8] += 1
        elif Y_value.between(17, 18).bool():
            count1[9] += 1
        elif Y_value.between(18, 19).bool():
            count1[10] += 1
print(count1)