def occurence(s):
    max=0
    for i in range(len(s)):
        count=1
        for j in range(i+1,len(s)):
            if s[i]==s[j]:
                count+=1
        if count>max:
            max=count
            char=s[i]
    return char,max
s=input("enter string:")
char,max=occurence(s)
print("character with maximum occurence :", char)
print("number of occurence :",max )