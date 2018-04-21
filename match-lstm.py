import create_embeddings
import get_ans_seq
import numpy as np
# print("start ....")
#paraMat, ques ,ans = create_embeddings.main()
ans = get_ans_seq.main()
import tensorflow as tf
losses = []
import gc

l = 25
P = 387
Q = 25
d = 50
count = 0

Batch_size = 1
dataPara = tf.placeholder(tf.float32, [Batch_size, P, d])
dataQues = tf.placeholder(tf.float32, [Batch_size, Q, d])

cell = tf.nn.rnn_cell.LSTMCell(l, state_is_tuple = True)

valPara, statePara = tf.nn.dynamic_rnn(cell, dataPara, dtype=tf.float32)
valQues, stateQues = tf.nn.dynamic_rnn(cell, dataQues, dtype=tf.float32)

# valPara = tf.transpose(valPara, [0, 2, 1])
# valQues = tf.transpose(valQues, [0, 2, 1])

Wq = tf.get_variable("Wq", shape = [l, l], initializer=tf.random_normal_initializer())
Wp = tf.get_variable("Wp", shape = [l, l], initializer=tf.random_normal_initializer())
Wr = tf.get_variable("Wr", shape = [l, l], initializer=tf.random_normal_initializer())
bp = tf.get_variable("bp", shape = [l, 1], initializer = tf.zeros_initializer)
b = tf.get_variable("b", shape = [1], initializer = tf.zeros_initializer)
w = tf.get_variable("w", shape = [l, 1], initializer = tf.zeros_initializer)
Wa = tf.get_variable("Wa", shape = [l, l], initializer=tf.random_normal_initializer())
V = tf.get_variable("V", shape = [l, 2*l], initializer=tf.random_normal_initializer())
ba = tf.get_variable("ba", shape = [l, 1], initializer = tf.zeros_initializer)
v = tf.get_variable("v", shape = [l, 1], initializer = tf.zeros_initializer)
c = tf.get_variable("c", shape = [1], initializer = tf.zeros_initializer)

stateHr = cell.zero_state(1, tf.float32)
total_valHr = []
rev_total_valHr = []
valHr = tf.Variable(tf.truncated_normal([l, 1], dtype = tf.float32))

batch_prob = 0 

for j in range(0,Batch_size):
   
    
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
    
    ann = ans[count].split(' ')
    prob = 1
    count = count + 1
    
    #annPred = []
    
    for k in range(0,P+1):
        temp1 = tf.matmul(V,hHr)
        Fk = tf.tanh(tf.tile(tf.matmul(Wa,tf.reshape(valHr,(l,1))) + ba , (1,P+1)) + temp1)
        betak = tf.nn.softmax(tf.matmul(tf.transpose(v), Fk) + tf.tile(tf.reshape(c, (1, 1)), (1,P+1)))
        valHr , stateHr = tf.nn.dynamic_rnn(cell, tf.reshape(tf.matmul(hHr,tf.transpose(betak)),(1,1,2*l)), initial_state = stateHr)
        
        betak = tf.reshape(betak,(P+1,1))
        if k < len(ann):
            prob = prob * betak[int(ann[k])]
        betak = []   
        
        #answerr prediction code
        
        # currPred = 0
        # while currPred < P :
        #     currPred = np.argmax(np.random.multinomial(1, betak.T[0], 1), 1)
        #     annPred.apend(currPred)
            
        # answer prediction end
            
            
        
    
    hHr =[]
    #ann = []

    batch_prob = batch_prob - (tf.log(prob))
    
    print("probability done .......")
    gc.collect()
optimizer = tf.train.AdamOptimizer()

minimize = optimizer.minimize(batch_prob)
print("session about to start..........")
init_op = tf.initialize_all_variables()
# with tf.Session() as sess:
#      with tf.device("/GPU:0"):
#          sess.run(init_op)
#      gc.collect()
     # print("start training .....")
#no_of_batches = int(len(paraMat)/Batch_size)
sess = tf.Session()
sess.run(init_op)
epoch = 5  
for i in range(epoch):
    for (paraMat , ques) in create_embeddings.main():
        paraMat = np.reshape(paraMat, (1,387, 50))
        ques = np.reshape(ques, (1,25,50))
        inp, quest = paraMat , ques
            
        print("I am here")
            
        sess.run(minimize,{dataPara: inp, dataQues :quest})
        #print(sess.run(Wq))
        # print()
        # print()
        # print(sess.run(Wp))
        # print()
        # print()
        # print(sess.run(Wa))
        # print()
        #print(sess.run(bp))
        # print()
        # print()
        #print(sess.run(c))


    print ("Epoch - ",str(i))
t_dir = "C:\\Users\\user\\Desktop\\project\\trained_model"
saver = tf.train.Saver()
saver.save(sess, "%s\\best_model.chk" %t_dir)
#incorrect = sess.run(error,{data: test_input, target: test_output})
         
#print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))
# sess.close()


        
