#! /bin/bash
rsync -ave ssh --exclude={'*.pyc','runs'} * ysloots@astroluiz.science.ru.nl:/home/ysloots/masterproject/project

