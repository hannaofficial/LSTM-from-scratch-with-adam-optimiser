import numpy as np

#here I gonna explain you each line 
#This is LSTM model from scratch I also add adamoptimiser for weight update even adagrad also present

class Data:
    def __init__(self,path, seq_length):
        self.fp = open(path,'r')
        self.data = self.fp.read()
        character = list(set(self.data))
        self.char_to_index = { ch:i for (i,ch) in enumerate(character)}
        self.index_to_char = { i:ch for (i,ch) in enumerate(character)}
        self.data_size = len(self.data)
        self.vocab_size = len(character)
        self.tracker = 0
        self.seq_length = seq_length

    def next_batch(self):
        input_start = self.tracker
        input_end = self.tracker + self.seq_length
        inputs = [self.char_to_index[ch] for ch in self.data[input_start:input_end]]
        targets = [self.char_to_index[ch] for ch in self.data[input_start+1 :input_end +1]]
        self.tracker += self.seq_length
        if self.tracker + self.seq_length >= self.data_size:
            self.tracker = 0

        return inputs, targets

    def reset_tracker(self):
        return self.tracker == 0

    def close_file(self):
        self.fp.close()


def init_orthogonal(param):

    if param.ndim < 2:
        raise ValueError("Only parameters with 2 or more dimension are supported")

    rows, cols = param.shape
    new_param  = np.random.randn(rows, cols)

    if rows < cols:
        new_param = new_param.T

    q, r = np.linalg.qr(new_param)

    d = np.diag(r,0)
    ph = np.sign(d)
    q *= ph

    if rows < cols:
        q = q.T
    new_param = q
    return new_param





def init_lstm(hidden_size, vocab_size, z_size):

    W_f = np.random.randn(hidden_size, z_size)
    b_f = np.zeros((hidden_size, 1))

    W_i = np.random.randn(hidden_size, z_size)
    b_i = np.zeros((hidden_size, 1))

    W_g = np.random.randn(hidden_size, z_size)
    b_g = np.zeros((hidden_size, 1))

    W_o =  np.random.randn(hidden_size, z_size)
    b_o = np.zeros((hidden_size, 1))

    W_v = np.random.randn(vocab_size, hidden_size)
    b_v = np.zeros((vocab_size, 1))

    W_i = init_orthogonal(W_i)
    W_f = init_orthogonal(W_f)
    W_g = init_orthogonal(W_g)
    W_o = init_orthogonal(W_o)
    W_v = init_orthogonal(W_v)

    params =  W_f, W_i, W_g, W_o, W_v, b_f, b_i, b_g, b_o, b_v
    return params

    


def one_hot_encode(input_idx,vocab_size):
    one_hot_encode = np.zeros(vocab_size)
    one_hot_encode[input_idx]=1
    return one_hot_encode

def one_hot_encode_sequence(provide_seq, vocab_size):
    
    encoding = np.array([one_hot_encode(idx,vocab_size) for idx in provide_seq])
    encoding = encoding.reshape(encoding.shape[0], encoding.shape[1], 1)
    return encoding
    
def clip_grad_norm(grads, max_norm=0.25):
    max_norm = float(max_norm)
    total_norm = 0

    for grad in grads:
        grad_norm = np.sum(np.power(grad , 2))
        total_norm += grad_norm

    total_norm = np.sqrt(total_norm)

    clip_coff = max_norm/(total_norm + 1e-6)

    if clip_coff < 1:
        for grad in grads:
            grad *= clip_coff

    return grads


def sigmoid(x, derivation=False):
    
    x += 1e-12
    f = 1 / (1 + np.exp(-x))
    
    if derivation: # Return the derivative of the function evaluated at x
        return f * (1 - f)
    else: # Return the forward pass of the function at x
        return f

def softmax(x, derivation=False):
    x += 1e-12
    f = np.exp(x)/ np.sum(np.exp(x))
    if derivation:
        pass
    else:
        return f

def tanh(x, derivation = False):
    x += 1e-12
    f = (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
    if(derivation):
        return 1 - f**2
    else:
        return f

def adagrad_optimiser(params,grads,  sq_grad, learning_rate = 1e-4):

    for param, grad, square_grad in zip(params, grads, sq_grad):
        square_grad += np.power(grad,2)  #adding variance without weightage can hider the adagrad to converge to o error as variance increase the learning rate decrease and decrease and it can converse to zero alse that can effect on update of weight
        param += -learning_rate*grad/np.sqrt(square_grad+1e-8)

    return params


def adam_optimiser(params, grads, sq_grad,moment_grad, learning_rate = 1e-4):
     beta = 0.9
     beta_2 = 0.999
     t=1 
     for param, grad, variance_grad, m_grad in zip(params, grads, sq_grad,moment_grad):
         variance_grad = beta_2*variance_grad + (1-beta_2)*np.power(grad,2) 
         v_hat = variance_grad / (1 - beta_2**t)
         m_grad = beta*m_grad + (1 - beta)*grad
         m_hat = m_grad / (1 - beta**t)
         param = param - m_hat*(learning_rate/(np.sqrt(v_hat) + 1e-8))  # below is actually root over of variace i.e. standard deviation it actually adjust the learning rate base on s.d
         t+=1


     return params


#forward logic

def forward(inputs, h_prev, C_prev, param):

    assert h_prev.shape == (hidden_size, 1)
    assert C_prev.shape == (hidden_size, 1)

    W_f, W_i, W_g, W_o, W_v, b_f, b_i, b_g, b_o, b_v = param

    x_s, z_s, f_s, i_s = [], [], [], []
    g_s, C_s, o_s, h_s = [], [], [], []
    v_s, output_s = [], []

    h_s.append(h_prev)
    C_s.append(C_prev)


    for x in inputs:

        z = np.row_stack((h_prev, x))
        z_s.append(z)

        f = sigmoid(np.dot(W_f,z) + b_f)
        f_s.append(f)

        i = sigmoid(np.dot(W_i, z) + b_i)
        i_s.append(i)

        g = np.tanh(np.dot(W_g, z) + b_g)
        g_s.append(g)

        C_prev = f*C_prev + i*g
        C_s.append(C_prev)

        o = sigmoid(np.dot(W_o, z) + b_o)
        o_s.append(o)

        h_prev = o*np.tanh(C_prev)
        h_s.append(h_prev)

        
        v = np.dot(W_v, h_prev) + b_v
        v_s.append(v)

        output = softmax(v)
        output_s.append(output)

    return z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, output_s 

#backward logic

def backward(z, f, i, g, C, o, h, v, outputs, targets, p ):
    W_f, W_i, W_g, W_o, W_v, b_f, b_i, b_g, b_o, b_v = p
    
    W_f_d = np.zeros_like(W_f)
    b_f_d = np.zeros_like(b_f)

    W_i_d = np.zeros_like(W_i)
    b_i_d = np.zeros_like(b_i)

    W_g_d = np.zeros_like(W_g)
    b_g_d = np.zeros_like(b_g)

    W_o_d = np.zeros_like(W_o)
    b_o_d = np.zeros_like(b_o)

    W_v_d = np.zeros_like(W_v)
    b_v_d = np.zeros_like(b_v)

    dh_next = np.zeros_like(h[0])
    dC_next = np.zeros_like(C[0])

    loss = 0

    for t in reversed(range(len(outputs))):

            loss += -np.mean(np.log(outputs[t])*targets[t])   #output is actually softmax value
            C_prev = C[t-1]
    
            dv = np.copy(outputs[t])
            dv[np.argmax(targets[t])] -= 1
    
            W_v_d += np.dot(dv, h[t].T)
            b_v_d += dv
    
            dh = np.dot(W_v.T, dv)
            dh += dh_next
            do = dh*np.tanh(C[t])
            do *= sigmoid(o[t], derivation=True)
    
            W_o_d += np.dot(do, z[t].T)
            b_o_d += do
    
            dC = np.copy(dC_next)
            dC += dh*o[t]*tanh(C[t], derivation = True)
            dg = dC*i[t]
            dg *= tanh(g[t], derivation=True)
    
            W_g_d += np.dot(dg, z[t].T)
            b_g_d += dg
    
            di = dC*g[t]
            di *= sigmoid(i[t], derivation=True)
            W_i_d += np.dot(di, z[t].T)
            b_i_d += di
    
            df = dC*C_prev
            df *= sigmoid(f[t], derivation=True)
            W_f_d += np.dot(df, z[t].T)
            b_f_d += df
    
            dz = (np.dot(W_f.T, df) + np.dot(W_i.T , di) + np.dot(W_g.T, dg) + np.dot(W_o.T, do))
            dh_prev = dz[:hidden_size, :]
            dC_prev = f[t]*dC
    
    grads= W_f_d, W_i_d, W_g_d, W_o_d, W_v_d, b_f_d, b_i_d, b_g_d, b_o_d, b_v_d

    grads = clip_grad_norm(grads)
    return loss, grads
               
def sample(h_prev,C_prev,params, inputs,hidden_size, n, temperature=1.0):
    
    assert h_prev.shape == (hidden_size, 1)
    assert C_prev.shape == (hidden_size, 1)
    inputs = inputs.reshape((vocab_len, 1))

    W_f, W_i, W_g, W_o, W_v, b_f, b_i, b_g, b_o, b_v = params

    x_s, z_s, f_s, i_s = [], [], [], []
    g_s, C_s, o_s, h_s = [], [], [], []
    v_s, output_s = [], []

    h_s.append(h_prev)
    C_s.append(C_prev)


    for t in range(n):

        z = np.row_stack((h_prev, inputs))
        z_s.append(z)

        f = sigmoid(np.dot(W_f,z) + b_f)
        f_s.append(f)

        i = sigmoid(np.dot(W_i, z) + b_i)
        i_s.append(i)

        g = np.tanh(np.dot(W_g, z) + b_g)
        g_s.append(g)

        C_prev = f*C_prev + i*g
        C_s.append(C_prev)

        o = sigmoid(np.dot(W_o, z) + b_o)
        o_s.append(o)

        h_prev = o*np.tanh(C_prev)
        h_s.append(h_prev)

        
        v = np.dot(W_v, h_prev) + b_v
        v_s.append(v)

        output = softmax(v/temperature)
        index = np.random.choice(range(vocab_len), p=output.ravel())
        inputs = np.zeros((vocab_len, 1))
        inputs[index] = 1
        output_s.append(index)
    return output_s        


def train_LSTM(data_reader, hidden_size,params, sq_grad, moment_grad):  #I tried to put temperature but it didn't work
    iter_num = 0
    threshold = 0.01

    smooth_loss = -np.log(1/data_reader.vocab_size)*data_reader.seq_length
    while (smooth_loss > threshold):
        if data_reader.reset_tracker():
            h = np.zeros((hidden_size, 1))
            c = np.zeros((hidden_size, 1))

        inputs, targets = data_reader.next_batch()
        inputs_one_hot = one_hot_encode_sequence(inputs, data_reader.vocab_size)
        targets_one_hot  = one_hot_encode_sequence(targets, data_reader.vocab_size)

        z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, outputs = forward(inputs_one_hot, h, c, params)
        loss, grads = backward(z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, outputs, targets_one_hot, params)
        params = adam_optimiser(params, grads,sq_grad, moment_grad)
        smooth_loss = smooth_loss*0.999 + loss*0.001
        h = h_s[-1]
        c = C_s[-1]
        if not iter_num%500:
            # , temperature = initial_temperature*(0.999**(initial_temperature/100)) this temperature is used to concenterated the probability distribution to more repetared word
            sample_ix = sample(h, c, params, inputs_one_hot[0],hidden_size,n=100 )
            sample_txt = ''.join(data_reader.index_to_char[ix] for ix in sample_ix)
            print("")
            print(f"Iteration: {iter_num}  Loss: {smooth_loss:.4f}")
            print(f"Sample: {sample_txt}")

        iter_num += 1
            
            
data_reader = Data('panchatantra.txt',seq_length=40)
vocab_len = data_reader.vocab_size
hidden_size = 512
z_size = vocab_len + hidden_size
params  = init_lstm(hidden_size, vocab_len, z_size) 
moment_grad = [np.zeros_like(p) for p in params]
sq_grad =    [np.zeros_like(p) for p in params]

train_LSTM(data_reader,hidden_size, params, sq_grad, moment_grad)