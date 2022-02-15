
def compare(file1, file2):
  f1 = open(file1, 'r')
  s1 = f1.readlines()
  f1.close()

  f2 = open(file2, 'r')
  s2 = f2.readlines()
  f2.close()

  counter = 0

  print(len(s1))

  for i in range(len(s1)):
    if s1[i] != s2[i]:
      #print(s1[i], s2[i])
      counter += 1

  print(counter)
  
  


compare('submission.csv',
         'compare/submission (5)1644938673.csv')


