import pickle

data1 = {
    'a' : [1,2,3],
    'b' : ['sfsd','dfsgdfgdf'],
    'c' : None
}

selfref_list = [14,5,6]

output = open('data.pkl','wb')

pickle.dump(data1,output)
pickle.dump(selfref_list,output,-1)

output.close()