FROM ubuntu:14.04

# Update OS with Php5, SSH
# don't know if all these php5 modules are needed
RUN apt-get update && \
apt-get install -y php5 libapache2-mod-php5  ssh \
php5-fpm php5-cli php5-mysqlnd php5-pgsql php5-sqlite php5-redis \
php5-apcu php5-intl php5-imagick php5-mcrypt php5-json php5-gd php5-curl && \
php5enmod mcrypt && \
rm -rf /var/lib/apt/lists/*

# Overwrite Apache2 conf
COPY skylight-local/docker/apache2.conf /etc/apache2

# Add skylight conf to sites available
COPY skylight-local/docker/skylight.conf /etc/apache2/sites-available

# Copy bash file to remove pid of Apache2
# see https://forums.docker.com/t/cant-restart-my-container-httpd-pid-8-already-running/67633
COPY skylight-local/docker/httpd-foreground.sh /usr/local/bin/

# Enable Apache mods rewrite and headers
RUN a2enmod rewrite headers

# Disable default site config
RUN a2dissite 000-default

# Enable skylight site config
RUN a2ensite skylight

# Expose port 80 in container
EXPOSE 80

# Run Apache2 as main process
CMD ["httpd-foreground.sh"]

# ssh -4 -N -f -L 9122:127.0.0.1:8080 dspace@lac-repo-test14.is.ed.ac.uk
# docker run -it -p 8080:80 -v /home/hrafn/PhpstormProjects/skylight:/var/html/www/skylight -v /home/hrafn/PhpstormProjects/skylight-local:/var/html/www/skylight-local eerc_skylight
