<?php
require('connect.php');
include('check.php');
extract($_POST);
$senha = md5($senha);
if(mysqli_query($con, "UPDATE `user` 
SET `senha` = '$senha',
WHERE `user`.`id_cliente` = '$user_id';")){
$msg = "Registro Alterado Com Sucesso";
}else{
    $msg = "Erro ao Alterar";
}
session_start();
$_SESSION['msg'] = $msg;
echo mysqli_error($con);
header("location:index.php");

    ?>