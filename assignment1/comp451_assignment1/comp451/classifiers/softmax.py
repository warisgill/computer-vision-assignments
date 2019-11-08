from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg, regtype='L2'):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength
    - regtype: Regularization type: L1 or L2

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization! Implement both L1 and L2 regularization based on the      #
    # parameter regtype.                                                        #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # for i, x_i in enumerate (X):
    #   #step 1
    #   score_vector = np.dot(x_i, W)
    #   # print("Step 1, Shapes of x_i, W, score_vector", x_i.shape, W.shape, score_vector.shape)
    #   #step 2
    #   u_probs = np.exp(score_vector)
    #   # print("Step 2, Shape of u_probs", u_probs.shape)
    #   #step 3
    #   numerators = u_probs
    #   denominator = 1 / np.sum(u_probs)
    #   probs = numerators * denominator #u_prob / np.sum(u_prob)
    #   # print("Step 3, Shapes of numerators, denominator, probs", numerators.shape,1,probs.shape)
    #   #step 4 
    #   local_loss = (-1) * np.log(probs[y[i]])
    #   # print("Step 4, Shape of local_loss: ", 1)
    #   loss +=  local_loss/X.shape[0]

    #   ################## Backward Pass ###########################s 
      
    #   # derivative at step 4 
    #   d_local_loss_d_local_loss = -1
    #   dloss_dprobsy_i =  np.multiply((1/probs[y[i]]) , d_local_loss_d_local_loss) 

    #   # derivative extra
    #   d_probyi_dselect = np.zeros_like(probs).reshape(probs.shape[0],1)
    #   d_probyi_dselect[y[i]] = 1
    #   d_probyi_dselect = np.multiply(d_probyi_dselect , dloss_dprobsy_i)
      
      


    #   # print("Check 4", d_probyi_dselect.shape, numerators.shape)

    #   # derivative at step 3
    #   numerators = np.reshape(numerators, (numerators.shape[0], 1))
    #   d_probs_wrt_numerator  = np.multiply(denominator , dloss_dprobsy_i)
    #   d_probs_wrt_denominator = np.multiply(numerators , dloss_dprobsy_i)

    #   # print("Shapes d_probs_wrt_n, d", d_probs_wrt_numerator, d_probs_wrt_denominator.shape)

    #   # print("Check 3", d_probs_wrt_denominator.shape)

    #   d_denominator_d_sum =  np.multiply((-1/((np.sum(probs))**2)), d_probs_wrt_denominator)
    #   # print("check 2", d_denominator_d_sum.shape )
    #   d_sum_d_e = np.multiply(np.ones_like(u_probs).reshape(u_probs.shape[0],1) , d_denominator_d_sum)
      
    #   # print("Check 1", d_sum_d_e.shape, d_probs_wrt_numerator.shape)
    #   u_probs = np.reshape(u_probs,(u_probs.shape[0],1))
    #   d_e_d_scores = np.multiply(u_probs , d_sum_d_e) +  np.multiply(u_probs , d_probs_wrt_numerator)
    #   # print("Check 0", d_e_d_scores.shape)

      
    #   d_scores_dw  =  np.multiply(np.ones_like(dW) , x_i[:,None])  #* d_e_d_scores 
      
    #   d_scores_dw = np.multiply(d_scores_dw , d_e_d_scores.T)
    #   # d_upstream_denominator = np.multiply((np.ones((10,1)) * (-1/(denominator**2))) , d_probs_wrt_denominator)
    #   # d_upstream_denominator = np.multiply(d_probs_wrt_denominator , (-1/(denominator**2))) 
    #   # d_upstream_numerator =  1 * d_probs_wrt_numerator

    #   # print("d_upstream_numerator, d_upstream_denominator", d_upstream_numerator, d_upstream_denominator.shape) 

    #   # derivative at step 2
    #   # d_upstream_u_probs = (u_probs * d_upstream_numerator) + np.multiply(u_probs,d_upstream_denominator)
      
    #   # print("Shape d_upstream_u_probs",d_upstream_u_probs.shape)
      
    #   # derivative at step 1
    #   # d_w = np.ones_like(dW)
    #   # d_w = (d_w * x_i[:,None])* d_upstream_u_probs   
    #   # print("Print check",d_w.shape)
    #   dW += (d_scores_dw/X.shape[0])


    #   # d_w  = np.multiply(np.multiply(np.ones_like(dW) , x_i) , d_upstream_u_probs)
    #   # dW += d_w    
    # # dW = dW/X.shape[0]
    
    for i, x_i in enumerate (X):
      scores = np.dot(x_i, W)
      exps = np.exp(scores)
      numerators = exps
      denominator = 1 / np.sum(exps)
      probs = numerators * denominator  
      local_loss = (-1) * np.log(probs[y[i]])
      loss +=  local_loss/X.shape[0]
      
      dscores = probs

      for k in range(W.shape[1]):
        if y[i] == k:
          dscores[k] -= 1
      dscores /= X.shape[0]
      # print("Check",x_i.shape,dscores.shape, dW.shape) 
      dw_temp = (np.multiply(np.ones_like(dW) , x_i[:,None])) * dscores
      dW += dw_temp
    
    regularization_loss  = 0.0
    if regtype == "L1":
      for i in range(W.shape[0]):
        for j in range(W.shape[1]):
          regularization_loss += W[i,j]
      dW += reg    
    else:
      for i in range(W.shape[0]):
        for j in range(W.shape[1]):
          regularization_loss += W[i,j] ** 2
      dW += reg * W

    loss = loss + reg * regularization_loss

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg, regtype='L2'):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization! Implement both L1 and L2 regularization based on the      #
    # parameter regtype.                                                        #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = np.dot(X,W)
    exps = np.exp(scores)
    probs =  exps* ((1/np.sum(exps,axis=1)).reshape(exps.shape[0],1))
    # print("check 1", probs.shape)
    # y.shape = (y.shape[0],1)
    Li = (-1) * np.log(probs[np.arange(probs.shape[0]), y])

    # print("check",Li.shape, y.shape, probs.shape)

    loss = np.sum(Li)/X.shape[0]  

    # print(probs[0,:])

    dscores = probs

    dscores[np.arange(probs.shape[0]), y] -=1 
    dscores /= X.shape[0]
    dW = np.dot(X.T, dscores)

    if regtype == "L1":
      regularization_loss  = np.sum(W)
      dW += reg    
    else:
      regularization_loss = np.sum(np.power(W,2))
      dW += reg * W

    loss = loss + reg * regularization_loss
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
