# Transcribr - handwritten text recognition

The app described here is up at http://transcribr.net. Test it out for yourself!

Test changes locally by installing Docker and using the following command:

```
docker build --rm -t transcribr-v1 . && docker run --rm -it -p 5000:5000 transcribr-v1
```

Much of the code here is based on the fastai guide for production deployment to Render: https://course.fast.ai/deployment_render.html.

<!-- 
--rm: Remove intermediate containers after a successful build
-t:   name and tag

-it:  creates an interactive bash shell in the container
-p:   Publish a container’s port(s) to the host
-v:   Bind mount a volume for hotreload    # -v /Users/adamschiller/Projects/DeepLearning/Transcribr-app/:/var/app
 -->