@echo off
echo Uploading Air Quality Index Prediction project to GitHub...

git init
git add .
git commit -m "Air Quality Index Prediction with high-accuracy models"
git branch -M main
git remote add origin https://github.com/duttasayan835/Skill4Future.git
git push -u origin main

echo Done! Check your GitHub repository at https://github.com/duttasayan835/Skill4Future
pause
