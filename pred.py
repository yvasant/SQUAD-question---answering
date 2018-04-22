import get_embedding
import numpy as np
print("start ....")
import tensorflow as tf
import gc

l = 25
P = 396
Q = 36
d = 50
count = 0

Batch_size = 1
dataPara = tf.placeholder(tf.float32, [Batch_size, P, d])
dataQues = tf.placeholder(tf.float32, [Batch_size, Q, d])
cell = tf.nn.rnn_cell.LSTMCell(l, state_is_tuple = True)

valPara, statePara = tf.nn.dynamic_rnn(cell, dataPara, dtype=tf.float32)
valQues, stateQues = tf.nn.dynamic_rnn(cell, dataQues, dtype=tf.float32)

## Define all the model parameters

Wq = tf.get_variable("Wq", shape = [l, l])
Wp = tf.get_variable("Wp", shape = [l, l])
Wr = tf.get_variable("Wr", shape = [l, l])
bp = tf.get_variable("bp", shape = [l, 1])
b = tf.get_variable("b", shape = [1])
w = tf.get_variable("w", shape = [l, 1])
Wa = tf.get_variable("Wa", shape = [l, l])
V = tf.get_variable("V", shape = [l, 2*l])
ba = tf.get_variable("ba", shape = [l, 1])
v = tf.get_variable("v", shape = [l, 1])
c = tf.get_variable("c", shape = [1])

annPred = []     

stateHr = cell.zero_state(1, tf.float32)
total_valHr = []
rev_total_valHr = []
beta = []
valHr = tf.Variable(tf.truncated_normal([l, 1], dtype = tf.float32))

batch_prob = 0
annPrediction = 0
betak = 0

for j in range(0,Batch_size):
    print("batch :",j)
    print("forward calculations......")
    for i in range(0,P) :
	    temp1 = tf.matmul(Wq,tf.reshape(valQues[j],[l,Q]))
	    G = tf.tanh(tf.tile(tf.matmul(Wp, tf.reshape(valPara[j][i], (l, 1))) + tf.matmul(Wr,tf.reshape(valHr ,(l,1))) + bp, (1, Q)) + temp1)
	    alpha = tf.nn.softmax(tf.matmul(tf.transpose(w), G) + tf.tile(tf.reshape(b, (1, 1)), (1,Q)))
	    z = tf.concat([tf.reshape(valPara[j][i], (l, 1)), tf.matmul(tf.transpose(valQues[j]), tf.transpose(alpha))], axis = 0)
	    valHr, stateHr = tf.nn.dynamic_rnn(cell, tf.transpose(tf.reshape(z, (1, 2 * l, 1)), (0, 2, 1)), initial_state = stateHr , dtype=tf.float32)
	    total_valHr.append(valHr)  
    print("calculations for rev....")
    gc.collect()
    for i in range(P-1,-1,-1):
        temp1 = tf.matmul(Wq,tf.reshape(valQues[j],[l,Q]))
        revG = tf.tanh(tf.tile(tf.matmul(Wp, tf.reshape(valPara[j][i], (l, 1))) + tf.matmul(Wr, tf.reshape(valHr ,(l,1))) + bp, (1, Q)) + temp1)
        alpha = tf.nn.softmax(tf.matmul(tf.transpose(w), revG) + tf.tile(tf.reshape(b, (1, 1)), (1,Q)))
        z = tf.concat([tf.reshape(valPara[j][i], (l, 1)), tf.matmul(tf.transpose(valQues[j]), tf.transpose(alpha))], axis = 0)
        valHr, stateHr = tf.nn.dynamic_rnn(cell, tf.transpose(tf.reshape(z, (1, 2 * l, 1)), (0, 2, 1)), initial_state = stateHr , dtype=tf.float32)
        rev_total_valHr.append(valHr)
    total_valHr = tf.reshape(total_valHr,(l,P))
    rev_total_valHr = tf.reshape(rev_total_valHr,(l,P))
    hHr = tf.concat([total_valHr,rev_total_valHr],axis=0)
    total_valHr = []
    rev_total_valHr = []
    tocon = tf.zeros([2*l,1])
    hHr = tf.concat([hHr,tocon],axis=1)
    gc.collect()
    print("betak calculations.....")
    #ann = ans[count].split(' ')
    prob = 1  
    count = count + 1
    
    for k in range(0,P+1):
        temp1 = tf.matmul(V,hHr)
        Fk = tf.tanh(tf.tile(tf.matmul(Wa,tf.reshape(valHr,(l,1))) + ba , (1,P+1)) + temp1)
        betak = tf.nn.softmax(tf.matmul(tf.transpose(v), Fk) + tf.tile(tf.reshape(b, (1, 1)), (1,P+1)))
        valHr , stateHr = tf.nn.dynamic_rnn(cell, tf.reshape(tf.matmul(hHr,tf.transpose(betak)),(1,1,2*l)), initial_state = stateHr)
        betak = tf.reshape(betak,(P+1,1))
        beta.append(betak)
            
    ann = []
    hHr =[]
    # annPrediction = annPred
    print("probability done .......")
    gc.collect()

print("session about to start..........")

gc.collect()

saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, "./trained_model/best_model.chk")
print("Model loaded")
for (paraMat, ques) in spy_anu_pred.main() :
    paraMat = np.reshape(paraMat, (1, 249, 50))
    ques = np.reshape(ques, (1, 19, 50))
    inp, quest = paraMat, ques
    print("Start predicting.........")
   
    prediction = sess.run(beta, {dataPara: inp, dataQues: quest})
    print("done with prediction........ ")
    #print(prediction)
prediction = np.around(prediction , decimals = 6)
 
prob_ans = []
for i in range(5):
    pred = 0
    predprev = 0
    idx = 0
    pred = 0
    predprev = 0
    arrPred = []
    while (pred < 249) and (idx < 249):
	    while (predprev >= pred) :
	        pred = np.argmax(np.random.multinomial(1, prediction[idx].T[0]))
	    predprev = pred
	    arrPred.append(pred)
	    idx = idx + 1

    prob_ans.append(arrPred)
print(prob_ans)
