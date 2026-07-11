list=[2,7,4,1,3,6]

print(list)
pairs=[]# separate list to store pairs 
def findpairs(list):
    for i in range(len(list)):
        a=list[i]
        for j in range(i+1,len(list)):
            if a+list[j]==10:
                pairs.append((a,list[j]))
            else:
                continue
    return pairs

print(" pairs are:",findpairs(list))

