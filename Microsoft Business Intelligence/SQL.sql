
 --UNION tow different tables 
 
	(SELECT name
		FROM Person
		WHERE City=“Seattle”)
	UNION
	(SELECT name
		FROM Person, Purchase
		WHERE buyer=name AND store=“The Bon”
		

-- Conserving Duplicates


		(SELECT name
			FROM Person
			WHERE City=“Seattle”)
		UNION ALL
		(SELECT name
			FROM Person, Purchase
			WHERE buyer=name AND store=“The Bon”)
			
			
			
--Subqueries

	SELECT Purchase.product
		FROM Purchase
		WHERE buyer =
			(SELECT name
				FROM Person
				WHERE ssn = ‘123456789‘);
   
   
   SELECT Purchase.product
		FROM Purchase, Person
		WHERE buyer = name AND ssn = ‘123456789‘;

  --Subqueries Returning Relations
  
	SELECT Company.name
		FROM Company, Product
		WHERE Company.name = Product.maker
				AND Product.name IN
	(SELECT Purchase.product
		FROM Purchase
		WHERE Purchase .buyer = ‘Joe Blow‘);
		
	
	SELECT Company.name
		FROM Company, Product, Purchase
		WHERE Company.name = Product.maker
				AND Product.name = Purchase.product
				AND Purchase.buyer = ‘Joe Blow’;
				
				
				

----Removing Duplicates

	SELECT DISTINCT Company.name
		FROM Company, Product
		WHERE Company.name= Product.maker
				AND Product.name IN
		(SELECT Purchase.product
				FROM Purchas
				Purchase.buyer = ‘Joe Blow’)
				
				
		SELECT DISTINCT Company.name
			FROM Company, Product, Purchase
			WHERE Company.name= Product.maker
			AND Product.name = Purchase.product
			AND Purchase.buyer = ‘Joe Blow’;


			
--Subqueries Returning Relations(s > ALL R; s > ANY R; EXISTS R 

		SELECT name
			FROM Product
			WHERE price > ALL (SELECT price
			FROM Purchase
			WHERE maker=‘Gizmo-Works’)
			

-- Coorelation Query 			
			
		SELECT DISTINCT title
			FROM Movie AS x
			WHERE year < > ANY
			(SELECT year
				FROM Movie
				WHERE title = x.title);

-- Complex Correlated Query


		SELECT DISTINCT pname, maker
			FROM Product AS x
			WHERE price > ALL (SELECT price
			FROM Product AS y
			WHERE x.maker = y.maker AND y.year < 1972);
			
			
--Existential/Universal Conditions


		SELECT DISTINCT Company.cname
			FROM Company, Product
			WHERE Company.cname = Product.company and Product.price < 100;
			
			
			SELECT DISTINCT Company.cname
				FROM Company
				WHERE Company.cname NOT IN (SELECT Product.company
				FROM Product
				WHERE Product.price >= 100;
				
--Aggregation
		SELECT Avg(price)
			FROM Product
			WHERE maker=“Toyota”;
			
	--Count
      SELECT Count(category) same as Count(*)
		FROM Product
		WHERE year > 1995;
		
		SELECT Count(DISTINCT category)
		FROM Product
		WHERE year > 1995;
		
--Grouping and Aggregation

	SELECT product, Sum(price*quantity) AS TotalSales
		FROM Purchase
		WHERE date > “9/1”
		GROUP BY product;
		
		SELECT product, Sum(price*quantity) AS TotalSales
			FROM Purchase
			WHERE date > “9/1”
			GROUP BY product;
			
			
--Nested Quereis

	SELECT DISTINCT x.product, (SELECT Sum(y.price*y.quantity)
		FROM Purchase y
		WHERE x.product = y.product
		AND y.date > ‘9/1’)
		AS TotalSales
		FROM Purchase x
		WHERE x.date > “9/1”;
		
		
   SELECT product, Sum(price * quantity) AS SumSales
			Max(quantity) AS MaxQuantity
			FROM Purchase
            GROUP BY product;
			
			
			
-- HAVING Clause

	SELECT product, Sum(price * quantity)
	FROM Purchase
		WHERE date > “9/1”
		GROUP BY product
		HAVING Sum(quantity) > 30;
		
-- NULL VALUE

	SELECT *
		FROM Person
		WHERE age < 25 OR age >= 25 OR age IS NULL

		
		
--OUTERJOIN

	SELECT Product.name, Purchase.store
	FROM Product JOIN Purchase ON
	Product.name = Purchase.prodName;
	
	SELECT Product.name, Purchase.store
		FROM Product, Purchase
		WHERE Product.name = Purchase.prodName;
		
		
		
		
--Insertions,Updates,Deletions


		INSERT INTO Product(name, listPrice)
			SELECT DISTINCT prodName, price
			FROM Purchase
			WHERE prodName NOT IN (SELECT name FROM Product);
			

			UPDATE PRODUCT
				SET price = price/2
				WHERE Product.name IN
				(SELECT product
				FROM Purchase
				WHERE Date =‘Oct, 25, 1999’);
				
				
				
				
		CREATE TABLE Person(
			name VARCHAR(30),
			social-security-number INT,
				age SHORTINT,
				city VARCHAR(30),
				gender BIT(1),
				Birthdate DATE
				);
				
				
				CREATE TABLE Person(
					name VARCHAR(30),
					social-security-number INT,
					age SHORTINT DEFAULT 100,
					city VARCHAR(30) DEFAULT ‘Seattle’,
					gender CHAR(1) DEFAULT ‘?’,
						Birthdate DATE;
						
						
						
						
						
	--Defining Views

	
	CREATE VIEW Developers AS
	SELECT name, project
		FROM Employee
	WHERE department = “Development”;
	
	

	SELECT name, Purchase.store
		FROM Person, Purchase, Product
		WHERE Person.city = “Seattle” AND
		Person.name = Purchase.buyer AND
		Purchase.poduct = Product.name AND
		Product.category = “shoes”;
		
		
		
--UPDATEING VIEW 

	CREATE VIEW Developers AS
		SELECT name, project
		FROM Employee
		WHERE department = “Development”;
		
		INSERT INTO Developers
		VALUES(“Joe”, “Optimizer”)

		
		INSERT INTO Employee(ssn, name, department, project, salary)
			VALUES(NULL, “Joe”, NULL, “Optimizer”, NULL)

			


