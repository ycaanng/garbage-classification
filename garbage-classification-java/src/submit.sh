#!/usr/bin/env bash
mvn clean package -DskipTests
rm -rf package*
mkdir package
mv ./target/garbage-classification-java-1.0-SNAPSHOT.jar ./package
zip -r package.zip ./package
