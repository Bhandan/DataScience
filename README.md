
###### This Data Science Practices project, to incress the expertise level #######

### TECHNOLOGY USED ### 
 		1.PySpark 
		2.Pandas
		3.R


#### NUMBER OF PROJECTS FOR PRACTICES ###

        1.Iris Data
		2.Titanic Data
		3.loan Predication Data
		4.Bigmart Sales Data
		5.Boston Housing Data
		6.Human Activety Recognitiion Data
		7.Black Friday Data
		8.Siam Competetion Data
		9.Trips Histry Data
		10.Million Songs Data
		11. Census Data
		12.Movie Lens Data
		13.Idenify your  Digits
		14. Yelp Data
		15.ImageNet Data
		16.KDD Cup 1998
		17.Chicago Crime Data
		18.NewYork Yellow Taxi Data
		19.NewYork Green Taxi Data

		 
### MODELS ARE USED ###
	
 	A. Supervised Learning
	B. Unsupervised
	C. Semi-supervised Learning 
	D. Reinforcement Learning
	
##### Machine Learning Algorithms: ####	
	
	1. Linear Regression
					
		A. Advantages
			1. Simple to Explain
			2. Highly Interpretable
			3. Model Training and Predicationare fast
			4. No tuning is required
			5. Features don't need scaling
			6. Can Perform Well with a small number of observation
			7. Well understood
		B. Disadvantages
			1. Presumes a linear relationship between the feature and the response
			2. Performance is not competitive
			3. Can't automatically learn feature interaction 
		
		#Python
					
			from sklearn import linear_model
			x_train=input_variables_values_training_datasets
			y_train=target_variables_values_training_datasets
			x_test=input_variables_values_test_datasets
			linear = linear_model.LinearRegression()
			linear.fit(x_train, y_train)
			linear.score(x_train, y_train)
			predicted= linear.predict(x_test)
		
		#R 
			x_train <- input_variables_values_training_datasets
			y_train <- target_variables_values_training_datasets
			x_test <- input_variables_values_test_datasets
			x <- cbind(x_train,y_train)
			linear <- lm(y_train ~ ., data = x)
			summary(linear)
			predicted= predict(linear,x_test) 

	2.Logistic Regression
		
		A. Advantages
			1. Highly interpretable
			2. Model training and prediction are fast
			3. No tuning is required
			4. Features don't need scaling
			5. Can perform well with a small number of observations
			6. Outputs well-calibrated predicted probabilities
		B. Disadvantages
			1. Presumes a linear relationship between the features and the log-odds of the response
			2. Performance is not competitive with the best supervised learning methods
			3. Can't automatically learn feature interactions
			
		#Python
			from sklearn.linear_model import LogisticRegression
			model = LogisticRegression()
			model.fit(X, y)
			model.score(X, y)
			predicted= model.predict(x_test)
		
		#R					
			x <- cbind(x_train,y_train)
			logistic <- glm(y_train ~ ., data = x,family='binomial')
			summary(logistic)
			predicted= predict(logistic,x_test)
					
	3. Decision Trees
		A. Advantages
			1. Can be used for regression or classification
			2. Can be displayed graphically
			3. Highly interpretable
			4. Can be specified as a series of rules, and more closely approximate human decision-making than other models
			5. Prediction is fast
			6. Features don't need scaling
			7. Automatically learns feature interactions
			8. Tends to ignore irrelevant features
			9. Non-parametric (will outperform linear models if relationship between features and response is highly non-linear)
		B. Disadvantages
			1. Performance is (generally) not competitive with the best supervised learning methods
			2. Can easily overfit the training data (tuning is required)
			3. Small variations in the data can result in a completely different tree (high variance)
			4. Recursive binary splitting makes "locally optimal" decisions that may not result in a globally optimal tree
			5. Doesn't tend to work well if the classes are highly unbalanced
			6. Doesn't tend to work well with very small datasets
			
		#Python			
			from sklearn import tree
			model = tree.DecisionTreeClassifier(criterion='gini')
			model.fit(X, y)
			model.score(X, y)
			predicted= model.predict(x_test)
		#R 
			library(rpart)
			x <- cbind(x_train,y_train)
			fit <- rpart(y_train ~ ., data = x,method="class")
			summary(fit)
			predicted= predict(fit,x_test)
							
	4.SVM(Support Vector Machine)
					
		#Python
			from sklearn import svm
			model = svm.svc()
			model.fit(X, y)
			model.score(X, y)
			predicted= model.predict(x_test)
				
		#R 
			library(e1071)
			x <- cbind(x_train,y_train)
			fit <-svm(y_train ~ ., data = x)
			summary(fit)
			predicted= predict(fit,x_test)		
					
	5. Naive Bayes
																
						[P(c/X)=P(x1/c)*P(x2/c)*........*P(xn/c)*P(c)]
		A. Advantages
			1. Model training and predication are very fast
			2. Somewhat interpretable
			3. No tuning is required
			4. Features don't need scaling
			5. Insensitive to irrelevant feature
			6. Perform better than logistic regression when training set is very small
		B. Disadvantages
			1. Predicted probabilities are not well-calibrated
			2. Correlated features can be problematic
			3. Can't handle negative features with Multinomial Naive Bayes
			4. Has a higher "asymptotic error" than logistic regression
			
		#Python			
			from sklearn.naive_bayes import GaussianNB
			model.fit(X, y)
			predicted= model.predict(x_test)	
		#R			
			library(e1071)
			x <- cbind(x_train,y_train)
			fit <-naiveBayes(y_train ~ ., data = x)
			summary(fit)
			predicted= predict(fit,x_test)
			
	6. KNN(K-Nearest Neighbors)	
		A. Advantages:
			1. Simple to understand and explain
			2. Model training is fast
			3. Can be used for classfication and regression
		
		B. Disadvantages:
			1. Must store all of the training data
			2. Predication phase can be slow when N is large-scale
			3. Sensitive to irrelevant
			4. Sensitive to the scale of the data
			5. Accuracy is not competitive with the best supervised learning methods
			
		#Python
			from sklearn.neighbors import KNeighborsClassifier
			KNeighborsClassifier(n_neighbors=6) # default value for n_neighbors is 5
			model.fit(X, y)
			predicted= model.predict(x_test)
		#R 
			library(knn
			x <- cbind(x_train,y_train)
			fit <-knn(y_train ~ ., data = x,k=5)
			summary(fit)
			predicted= predict(fit,x_test)
					
	7. K-Means

		#Python
			from sklearn.cluster import KMeans
			k_means = KMeans(n_clusters=3, random_state=0)
			model.fit(X)
			predicted= model.predict(x_test)	
		#R 
			library(cluster)
			fit <- kmeans(X, 3) # 5 cluster solution
				
	8. Random Forests
		A. Advantages(compared to decision trees):
			1. Performance is competitive with the best supervised learning methods
			2. Provides a more reliable estimate of feature importance
			3. Allows you to estimate out-of-sample error without using train/test split or cross-validation
		B. Disadvantages
			1. Less interpretable
			2. Slower to train
			3. Slower to predict
			
		#Python
			from sklearn.ensemble import RandomForestClassifier
			model= RandomForestClassifier()
			model.fit(X, y)
			predicted= model.predict(x_test)
					
	9.Dimensionality Reduction Algorithms
	
		#Python
			from sklearn import decomposition
			train_reduced = pca.fit_transform(train)
			test_reduced = pca.transform(test)
		#R 
			library(stats)
			pca <- princomp(train, cor = TRUE)
			train_reduced  <- predict(pca,train)
			test_reduced  <- predict(pca,test)
			
	10.Gradient Boosting Algorithms
		
		10.1. GBM
				
			#Python
				from sklearn.ensemble import GradientBoostingClassifier
				model= GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
				model.fit(X, y)
				predicted= model.predict(x_test)
				
			#R 
				library(caret)
				x <- cbind(x_train,y_train)
				fitControl <- trainControl( method = "repeatedcv", number = 4, repeats = 4)
				fit <- train(y ~ ., data = x, method = "gbm", trControl = fitControl,verbose = FALSE)
				predicted= predict(fit,x_test,type= "prob")[,2]
				
		10.2. XGBoost
			
			#Python
				from xgboost import XGBClassifier
				from sklearn.model_selection import train_test_split
				from sklearn.metrics import accuracy_score
				X = dataset[:,0:10]
				Y = dataset[:,10:]
				seed = 1
				X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=seed)
				model = XGBClassifier()
				model.fit(X_train, y_train)
				y_pred = model.predict(X_test)
				
			#R 
				require(caret)
				x <- cbind(x_train,y_train)
				TrainControl <- trainControl( method = "repeatedcv", number = 10, repeats = 4)
				model<- train(y ~ ., data = x, method = "xgbLinear", trControl = TrainControl,verbose = FALSE)
				predicted <- predict(model, x_test)
				
				
		10.3. LightGBM
				a.Faster training speed and higher efficiency
				b.Lower memory usage
				c.Better accuracy
				d.Parallel and GPU learning supported
				e.Capable of handling large-scale data
			#Python 
				data = np.random.rand(500, 10) 
				label = np.random.randint(2, size=500)
				train_data = lgb.Dataset(data, label=label)
				test_data = train_data.create_valid('test.svm')
				param = {'num_leaves':31, 'num_trees':100, 'objective':'binary'}
				param['metric'] = 'auc'
				num_round = 10
				bst = lgb.train(param, train_data, num_round, valid_sets=[test_data])
				bst.save_model('model.txt')
				data = np.random.rand(7, 10)
				ypred = bst.predict(data)		
			#R 
				library(RLightGBM)
				data(example.binary)
				num_iterations <- 100
				config <- list(objective = "binary",  metric="binary_logloss,auc", learning_rate = 0.1,
						num_leaves = 63, tree_learner = "serial", feature_fraction = 0.8, bagging_freq = 5, 
						bagging_fraction = 0.8, min_data_in_leaf = 50, min_sum_hessian_in_leaf = 5.0)
				handle.data <- lgbm.data.create(x)
				lgbm.data.setField(handle.data, "label", y)
				handle.booster <- lgbm.booster.create(handle.data, lapply(config, as.character))
				lgbm.booster.train(handle.booster, num_iterations, 5)
				pred <- lgbm.booster.predict(handle.booster, x.test)
				sum(y.test == (y.pred > 0.5)) / length(y.test)
				lgbm.booster.save(handle.booster, filename = "/tmp/model.txt")
				
				
		10.4 Catboost
			#R 
				set.seed(1)
				require(titanic)
				require(caret)
				require(catboost)
				tt <- titanic::titanic_train[complete.cases(titanic::titanic_train),]
				data <- as.data.frame(as.matrix(tt), stringsAsFactors = TRUE)
				drop_columns = c("PassengerId", "Survived", "Name", "Ticket", "Cabin")
				x <- data[,!(names(data) %in% drop_columns)]y <- data[,c("Survived")]
				fit_control <- trainControl(method = "cv", number = 4,classProbs = TRUE)
				grid <- expand.grid(depth = c(4, 6, 8),learning_rate = 0.1,iterations = 100, l2_leaf_reg = 1e-3, 
						rsm = 0.95, border_count = 64)
				report <- train(x, as.factor(make.names(y)),method = catboost.caret,verbose = TRUE,
						preProc = NULL,tuneGrid = grid, trControl = fit_control)
				print(report)
				importance <- varImp(report, scale = FALSE)
				print(importance)



