#####################
#Starter code for Q3#
#####################

def equity(f):
    #inp is a dictionary key is company, value is a list [recievalbe, payable]
    receive = 0
    pay = 0
    for i in f:
        receive += f[i][0]
        pay += f[i][1]
    total = receive-pay
    return (receive,pay,total)

