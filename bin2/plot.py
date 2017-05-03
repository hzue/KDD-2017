import file_handler as fh

ans = fh.read_conclusion_file('res/conclusion/testing_ans.csv')
pred = fh.read_conclusion_file('./submit.csv')

print(ans)
