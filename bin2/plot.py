import util

test_start_date = '2016-10-11'
test_end_date = '2016-10-17'

ans = util._read_file('res/conclusion/testing_ans.csv')
pred = util._read_file('./result/0504_0188/submit.csv')

print(ans)
