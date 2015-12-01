
[TOC]


Ch 6. Feedforward Deep Networks
===================


 - a.k.a.  multilayer perceptrons (MLPs)
 - parametric functions deﬁned by composing together many parametric functions
 - multiple inputs and multiple outputs


----------


**terminology**

 - a ***layer*** of the network: each sub-function
 - a ***unit*** (or a ***feature***): each scalar output of one of these functions
![enter image description here](https://upload.wikimedia.org/wikipedia/en/5/54/Feed_forward_neural_net.gif)



**width and depth of a machine learning model**

 - ***width*** : the number of units in each layer
 - ***depth*** : the number of layer
 


 - conceptually simple example of an algorithm that captures the many advantages that come from having significant width and depth. 
 - the key technology underlying most of the contemporary commercial applications of deep learning to large datasets.

 - traditional machine learning algorithms (linear regression, linear classifiers, logistic regression and kernel machines): the functions are non-linear in the space of inputs x, but they are linear in some other pre-defined space.
 
 - Neural networks allow us to learn new kinds of non-linearity: 
 to learn the features provided to a linear model

----------


6.1 Vanilla (Shallow) MLPs
-------------
 - among the ﬁrst and most successful learning algorithms
 (Rumelhart et al., 1986e,c)
 - learn at least one function deﬁning the features, as well as a (typically linear) function mapping from features to output
 - hidden layers: 
 the layers of the network that correspond to features rather than outputs.
 
 - a ***vanilla MLP***:  architecture with a ***single*** hidden layer
 ![enter image description here](https://lh3.googleusercontent.com/OF3UmkAbqhapN5gOQPpsJ2KtfCJDG3BWNkklTe-vDlw=s0 "Screen Shot 2015-08-31 at 9.18.49 PM.png")
 - hidden unit vector: $h=sigmoid(c+Wx)$
 - output vector:  $\hat { y } =b+Vh$
 
----------
####Example 6.1.1. Vanilla (Shallow) Multi-Layer Neural Network for Regression
 - the input-output functions :
$$
{ f }_{ \theta  }(x)= b+Vsigmoid(c+Wx)\\ \\where \\ \\ sigmoid(a) = 1/(1+{ e }^{ -a })
\\
$$

 - input: $x\in { \Re  }^{ { n }_{ i } }$
 - hidden layer output: 
 $h =sigmoid(c+Wx)$
 
 - parameters: $\theta =(b,c,V,W)$
 - weight matrices: $V\in { \Re  }^{ { n }_{ o }\times { n }_{ h } }$ , $W\in { \Re  }^{ { n }_{ h }\times { n }_{ i } }$
 
 - loss function: $L(\hat { y } -y)= { \left\| \hat { y } -y \right\|  }^{ 2 }$

 - ${ L }^{ 2 }$ decay (regularizer): ${ \left\| \omega  \right\|  }^{ 2 }\quad =\quad \left( \sum _{ ij }{ { W }_{ ij }^{ 2 } } +\sum _{ ki }{{ W } _{ ki }^{ 2 }} \right) $
 
 - minimize by:
 $$
 J(\theta )=\lambda { \left\| \omega  \right\|  }^{ 2 }+\frac { 1 }{ n } \sum _{ l=1 }^{ n }{ { \left\| { y }^{ (t) }-(b+Vsigmoid(c+W{ x }^{ (t) })) \right\|  }^{ 2 } } 
 $$
 
 - training procedure (stochastic gradient descent):
 $$
\omega \quad \leftarrow \quad \omega \quad -\epsilon \left( 2\lambda +{ \nabla  }_{ \omega  }L({ f }_{ \theta  }({ x }^{ (t) },{ y }^{ (t) }) \right) \\ \beta \quad \leftarrow \quad \beta \quad -\epsilon { \nabla  }_{ \beta  }L\left( { f }_{ \theta  }({ x }^{ (t) }),{ y }^{ (t) } \right) 
 $$
 ----------
 
 - MLPs can learn powerful non-linear transformations: in fact, with enough hidden units they can represent arbitrarily complex but smooth functions, they can be universal approximators (Section 6.5). 
 - This is achieved by composing simple but non-linear learned transformations.
 


 - By transforming the data non-linearly into a new space, a classification problem that was not linearly separable (not solvable by a linear classifier) can become separable, as illustrated in Figures 6.2 and 6.3.
 
![enter image description here](https://lh3.googleusercontent.com/FaWeDeUwADxj5bqPvb9EH-DyL3toquiUFIN27QLF4us=s0 "Screen Shot 2015-08-31 at 8.30.31 PM.png")

![enter image description here](https://lh3.googleusercontent.com/vXtj5jxocQ8zbml_u8QWYsALFIKxZS5NbFsFq1Sp_Ss=s0 "Screen Shot 2015-08-31 at 8.30.35 PM.png")

----------

6.2 Estimating Conditional Statistics
-------------
- linear regression : any function f by defining the mean squared error of f
$$
E[{ ||y−f(x)|| }^{ 2 }]
$$
- generalization: minimizing it yields an estimator of the conditional expectation of the output variable y given the input variable x
$$
E_{ p(x,y) }\left[ { \left\| y-f(x) \right\|  }^{ 2 } \right] =E_{ p(x,y) }\left[ y|x \right] 
$$

- generalize conditional maximum likelihood (introduced in Section 5.6.1) to other distributions than the Gaussian

----------


6.3 Parametrizing a Learned Predictor
-------------
####6.3.1 Family of Functions
 - **motivation**: to compose *simple transformations* in order to obtain 
*highly non-linear* ones
 - (MLPs compose affine transformations and element-wise non-linearities)
 - hyperbolic tangent activation functions:
 $$
 { h }^{ k }=tanh({ b }^{ k }+{ W }^{ k }{ h }^{ k-1 })
$$
 - the input of the neural net: ${ h }^{ 0 }=x$
 - theoutputofthe k-th hidden layer: ${ h }^{ k }$

 - affine transformation $a = b+Wx$ \, elementwise
$$
h=\phi (a)⇔{ h }_{ i }=\phi ({ a }_{ i })=\phi ({ b }_{ i }+{ W }_{ i,: }x)
$$

 - non-linear neural network activation functions:
 ######Rectifier or rectified linear unit (ReLU) or positive part
 ######Hyperbolic tangent
 ######Sigmoid
 ######Softmax
 ######Radial basis function or RBF
 ######Softplus
 ######Hard tanh
 ######Absolute value rectification
 ######Maxout

 
 - the structure (also called architecture) of the family of input-output functions can be varied in many ways: 
*convolutional networks*, 
*recurrent networks*


####6.3.2 Loss Function and Conditional Log-Likelihood
 - In the 80’s and 90’s the most commonly used loss function was the squared error
$$
L({ f }_{ θ }(x),y)={ ||fθ(x)−y|| }^{ 2 }
$$
 
 
 - if f is unrestricted (non- parametric),
$$
 f(x) = E[y | x = x]
$$

 - Replacing the squared error by an absolute value makes the neural network try to estimate not the conditional expectation but the conditional median
 
 - **cross entropy objective function**: when y is a discrete label, i.e., for classification problems, other loss functions such as the Bernoulli negative log-likelihood4 have been found to be more appropriate than the squared error. ($y∈{ \left\{ 0,1 \right\}  }$)

$$
L({ f }_{ θ }(x),y)=−ylog{ f }_{ θ }(x)−(1−y)log(1−{ f }_{ θ }(x))
$$

- ${f}_{\theta}(x)$ to be strictly between 0 to 1: use the sigmoid as non-linearity for the output layer(matches well with the binomial negative log-likelihood cost function)




#####Learning a Conditional Probability Model
- loss function as corresponding to a conditional log-likelihood, i.e., the negative log-likelihood (NLL) cost function
$$
{ L }_{ NLL }({ f }_{ \theta  }(x),y)=−logP(y=y|x=x;θ)
$$
- example) if y is a continuous random variable and we assume that, given x, it has a Gaussian distribution with mean ${f}_{θ}$(x) and variance ${\sigma}^{2}$
$$
−logP(y|x;θ)=\frac { 1 }{ 2 } { ({ f }_{ \theta  }(x)−y) }^{ 1 }/{ σ }^{ 2 }+log(2π{ σ }^{ 2 })
$$
- minimizing this negative log-likelihood is therefore equivalent to minimizing the squared error loss.

- for discrete variables, the binomial negative log-likelihood cost func- tion corresponds to the conditional log-likelihood associated with the Bernoulli distribution (also known as cross entropy) with probability $p = {f}_{θ}(x)$ of generating y = 1 given x =$ x$
$$
{L}_{NLL}=−logP(y|x;θ)={−1}_{y=1}{logp−1}_{y=0}log(1−p)\\ =−ylog{f}_{θ}(x)−(1−y)log(1−{f}_{θ}(x))
$$

#####Softmax
- designed for the purpose of specifying multinoulli distributions:
$$
p=softmax(a)\Longleftrightarrow { p }_{ i }=\frac { { e }^{ { a }_{ i } } }{ \sum { _{ j }^{  }{ { e }^{ { a }_{ j } } } }  } 
$$
- consider the gradient with respect to the scores $a$.
$$
\frac { ∂ }{ ∂{ a }_{ k } } { L }_{ NLL }(p,y)=\frac { ∂ }{ ∂{ a }_{ k } } (−log{ p }_{ y })=\frac { ∂ }{ ∂{ a }_{ k } } ({ −a }_{ y }+log\sum _{ j }^{  }{ { e }^{ { a }_{ j } } } )\\ ={ −1 }_{ y=k }+\frac { { e }^{ { a }_{ k } } }{ \sum _{ j }^{  }{ { e }^{ { a }_{ j } } }  } ={ p }_{ k }-{1}_{y=k}
$$
or
$$
\frac { ∂ }{ ∂{ a }_{ k } } { L }_{ NLL }(p,y)=(p-{e}_{y})
$$
####6.3.3 Cost Functions For Neural Networks
- a good choice for the criterion is maximum likelihood regularized with dropout, possibly also with weight decay.

####6.3.4 Optimization Procedure
- a good choice for the optimization algorithm for a feedforward network is usually stochastic gradient descent with momentum.


6.4 Flow Graphs and Back-Propagation
-------------
**back-propagation**
- it just means the method for computing gradients in such networks
- the output of the function to differentiate (e.g., the training criterion J) is a scalar and we are interested in its derivative with respect to a set of parameters (considered to be the elements of a vector θ), or equivalently, a set of inputs
- The partial derivative of J with respect to θ (called the gradient) tells us whether θ should be increased or de- creased in order to decrease J

- the partial derivative of the cost J with respect to parameters θ can be *decomposed recursively* by taking into consideration the composition of functions that relate θ to J , via intermediate quantities that mediate that influence, e.g., the activations of hidden units in a deep neural network.

####6.4.1 Chain Rule
- the locally linear influence of a variable x on another one y
- output: the cost, or objective function, $z = J(g(θ))$
####6.4.2 Back-Propagation in an MLP

####6.4.3 Back-Propagation in a General Flow Graph

####6.4.4 Symbolic Back-propagation and Automatic Differentiation

####6.4.5 Back-propagation Through Random Operations and Graphical Models



6.5 Universal Approximation Properties and Depth
-------------

- feedforward networks with hidden layers provide a universal approximation framework.
- universal approximation theorem (Hornik et al., 1989; Cybenko, 1989) states that a feedforward network with a linear output layer and at least one hidden layer with any “squashing” activation function (such as the logistic sigmoid activation function) can approximate any Borel measurable function from one finite-dimensional space to another with any desired non-zero amount of error, provided that the network is given enough hidden units.



6.6 Feature / Representation Learning
-------------
-  limitations of convex optimization problem on the representational capacity: many tasks, for a given choice of input representation x (the raw input features), cannot be solved by using only a linear predictor.

- solutions: kernel machine, manually engineer the representation or features φ(x),  or learn the features.
 
6.7 Piecewise Linear Hidden Units
-------------

- 

- 
> Written with [StackEdit](https://stackedit.io/).
