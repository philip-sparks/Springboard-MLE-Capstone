## Machine Learning-based Email Classifier Backend API Program

This is an end-to-end implementation of an email classifier. Given a new email, a pyspark ml classification model will
assess which folder a user should place their email in, giving them automatic organization.

![](helper_images/pipeline.png)

## Project Instructions 

### Getting Started

1. Clone the repository, and navigate to the downloaded folder.
```
git clone https://github.com/philip-sparks/Springboard-AI/email_capstone/
```

2. Download dataset.

	  - [Kaggle mirror](https://www.kaggle.com/wcukierski/enron-email-dataset/version/1)
    ```
    1. Put downloaded data into `data` folder
    2. Unzip to show an `emails.csv` file
    ```
    - [Tagged emails](http://www.d.umn.edu/~tpederse/enron.html)
    
    ```
    1. Place this data into data folder
    2. Convert all .cd3 files into .csv
    3. Rest of the project should stay the same
    ```
3. Setup your Amazon S3 buckets.
  - [Amazon S3](https://s3.console.aws.amazon.com/s3/)
    ```
    1. Select "Create bucket", with a unqiue name, but no special configuration
    2. Create the following four folders: `emails_tagged`, `emails_nontagged`, `pipeline`, `trained_model`
    3. Place the Kaggle mirror `emails.csv` file within `emails_nontagged`
    4. Place the tagged email folders within `emails_tagged`
    ```

4. Create an Amazon EMR instance.
  - [Amazon EMR](https://console.aws.amazon.com/elasticmapreduce/)
    ```
    1. Select "Create cluster"
    2. Set the s3 folder to the location of your bucket.
    3. Use the Spark application with any emr-5.2x release.
    4. Set your instance type to m5.xlarge
    5. Use a PEM key for your EC2 key pair. You cannot SSH in without one.
    6. Select "Create cluster" again
    ```
5. Change AWS Security Group settings.
  - [Amazon EC2 Security Groups](https://console.aws.amazon.com/ec2/home?region=us-east-1#SecurityGroups)
    ```
    1. Select your ElasticMapReduce-master node
    2. On the "Inbound" tab, select `Edit`
    3. Add a rule type of SSH at 22 for a source of 0.0.0.0/0 to open the SSH tunnel
    4. Add a rule type of HTTP at 80 for a source of 0.0.0.0/0 to open the homepage.
    5. Add a rule type of TCP at 1234 for a source of 0.0.0.0/0 to open the /predict API 
    ```
6. Install Filezilla and SSH into your EC2-EMR instance
    ```
    1. File -> Site Manager... -> New site
    2. Options: Protocol = SSH, Host = ec2 URL (e.g. ec2-54-160-243-194.compute-1.amazonaws.com)
    3. Options (cont.): Logon Type = Key file, User = hadoop, Key file = AWS created key.pem
    4. Hit "Connect"
    5. Copy code folder into EC2 box
    ```
7. SSH and setup model
    ```
    1. Login: ssh -i ~/Downloads/sb-test-1.pem hadoop@ec2-34-229-64-64.compute-1.amazonaws.com
    2. cd email_prod_folder; sudo pip install -r requirements.txt
    3. python preprocessing.py
    4. python train.py
    5. python unittests.py
    - Your model has been trained, tested and is now ready for production.
    ```
8. Start the app.py file and test.
```
    1. python app.py
    2. Go to your EC2 instance at port 1234. e.g.http://ec2-54-160-243-194.compute-1.amazonaws.com:1234/
    3. You'll be given a welcome message and instructions for accessing the model.
    4. Open another terminal and type the following command:curl -H "Content-Type: application/json" -d '{"Body" : "Your Message"}' http://your-ec2-instance.compute.amazonaws.com:1234/predict
    5. If you received a prediction back, congratulations! That's what the proceeding 35 were for.
```

## FAQ

#### How do I log or create a log script?

Spark logging is within the S3 bucket.

#### Check this statement:

The Flask approach works, but it is not scalable. Another service, such as Kafka or Kinesis, would be needed to handle thousands of requests
at a time.

#### Where should we put preprocessing/model/test files?