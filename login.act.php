<?php
    include('connect.php');
    include('check.php');

    extract($_POST);
    $email = $_POST['email'];
    $senha = md5($senha);

    if($conta = mysqli_query($con, "SELECT * FROM `cliente` WHERE `email` = '$email' && `senha` = '$senha' LIMIT 1")){
        mysqli_query($con, "UPDATE `cliente` SET `online` = now() WHERE `email` = '$email' && `senha` = '$senha'");
        $user = mysqli_fetch_array($conta);
        setcookie("ID",$user['id'], time() + (86400 * 30), '/');
        header('location:api.php');
    }else{
        echo "Falha ao fazer login";
    }
?>