1. Infraestructura y Máquinas
    - La maquina de desarrollo es una maquina suse15 con la ip 172.30.188.137/26 , usuario --> agent passwd --> gal1$LEO , el repo de git se encuentra en https://git.indra.es/git/GALTTCMC/GALTTCMC y la branch es WORKING_G2G_DEVELOPMENT.
    - La VM de destino ya se encuentra desplegada, es accesible por ssh con el usuario --> root passwd --> root y la ip 172.130.188.147/26.
    - SonarQube se ecuentra en una maquina separada, en la maquina de desarrolo existe un script que lanza el proceso
2. Repositorio Git
    - El repo de git se encuentra en https://git.indra.es/git/GALTTCMC/GALTTCMC y la branch es WORKING_G2G_DEVELOPMENT
    - Los tags tienen este formato MAC_X_VXX_XX_XX_XX para versiones oficiales y VXX_XX_XX_XX para versiones internas, ambas tienen que ser monitorizadas.
3. Proceso de Compilación
    - En el home del usuario agent estara bajado el repositorio, cuando se detecte un cambio se tendra que copiar ese contenido a una carpeta de ese mismo home, dar permisos de ejecucion a los .sh "usando algo similar a find" y dentro de esa carpeta se lanza Development_TTCF/ttcf/utils/dvds/build_DVDs.sh , ese script compila todo lo necesario y crea los DVD's en la ruta base de esa carpeta. Como Ejemplo podemos decir que /home/agent/GALTTCMC contiene la copia del repo de git y /home/agent/compile contiene la copia de GALTTCMC pero con permisos de ejecucion, dentro de esa carpeta se ejecuta Development_TTCF/ttcf/utils/dvds/build_DVDs.sh y cuando acabe dentro de /home/agent/compile aparecen varios DVD's pero solo nos interesa uno con el nombre InstallationDVD.iso. 
    - No hace falta añadir depencias, todo esta en la branch que se clona de git
4. SonarQube
    - la URL de sonnar sería https://sonarqube.indra.es
    - Ya existe un proyecto creado GALTTCMC
    - Los umbrales serían coverage: ">= 80%"bugs: "= 0"vulnerabilities: "= 0"code_smells: "<= 10"security_hotspots: "= 0"
    - Como dato opcional tengo un sonar-project.properties que contiene: sonar.projectKey ,sonar.sources ,sonar.java.binaries ,sonar.java.libraries ,sonar.coverage.jacoco.xmlReportPaths ,sonar.cfamily.build-wrapper-output ,sonar.coverage.exclusions ,sonar.exclusions ,sonar.host.url ,sonar.login
5. Despliegue en VM
    -  El nombre de la maquina es: Releases
    -  La ip es: 172.130.188.147/26
    - credenciales: user --> root passwd --> root 
    - La maquina se encuntra configurada ya con IP, cpu, ram, lector de dvd, lo unico que hay que hacer es, una vez pasado sonnar y generado el DVD, ese dvd hay que subirlo a al datastore del vcenter, despues configurarlo en la maquina virtual, encender la maquina, montar el disco en /mnt/cdrom, crear en root la carpeta /root/install, copiar el contenido del dvd en /root/install, entrar en root/install y ejecutar install.sh. 
    - si quieres el .5 lo podemos ver mas adelante o cuando desarrollemos este punto.
6. Configuracion 
    - Prefiero crear un ci_cd_config.yaml separado.
    - Intervalo de polling que sea cada 5 minutos.
7. Notificaciones
    - Tengo dos opciones en mente y las dos deben ejecutarse en la maquina de desarrollo, la idea es usar tanto un script en /etc/profile.d/informacion.sh ( ya que hay usuarios que no estan coectados) y comando wall para los usuarios que esten conectados.
    la notificación podria ser algo asi:
    sudo wall <<EOF
    > #################################################
    > #                                               #
    > #   ¡ATENCIÓN: Nueva versión disponible!        #
    > #                                               #
    > #   Puedes clonar la nueva maquina o revisar    #
    > #   el informe de sonnar para más detalles.     #
    > #                                               #
    > #################################################
    > EOF
