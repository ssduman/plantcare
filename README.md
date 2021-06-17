# PlantCare # 
* Web application for leafs: https://bbm416-plant.herokuapp.com/
## Specifications: ##
* Predicts leaf type, whether infected or not and location of leafs in 10 seconds.
* Recognizes apple, blueberry, cherry, corn, grape, orange, peach, pepper bell, potato, raspberry, soybean, squash, strawberry and tomato.
## Dependencies: ##
* PyTorch
* [YoloV5](https://github.com/ultralytics/yolov5)
   - modified, some imports are deleted to fit to Heroku.
* OpenCV
* Django
* Pillow
## Run: ##
* Clone the repository, run `pip install -r requirements.txt`, `python manage.py migrate`, `python manage.py makemigrations` then `python manage.py runserver`
## Datasets: ##
* PlantVillage
* PlantDoc
## Images: ##
<table>
    <tr>
        <td align="center">
            <img src="https://github.com/ssduman/plantcare/blob/master/img/website.png" alt="home-page" width="384" height="216">
            <br />
            <i> home page of the site </i>
        </td>
        <td align="center">
            <img src="https://github.com/ssduman/plantcare/blob/master/img/prediction.png" alt="prediction" width="384" height="216">
            <br />
            <i> prediction of given image </i>
        </td>
    </tr>
</table>

### Bugs and Limitations: ###
* Due to Heroku, CPU version of PyTorch is used, so it is little slow.
