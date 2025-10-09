<?php

if(!$con = mysqli_connect('localhost', 'root', '' , 'jaloxa')){
    echo "Erro ao acessar a base de dados!";
}
mysqli_query($con,"SET NAMES utf8");

?>