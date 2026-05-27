print("enter the four marks")
b=0
for i in range(0,4):
    a=int(input("Enter:"))
    b=b+a
avg=b/4
if (avg <40):
    print('failed')
else:
    print("passed")