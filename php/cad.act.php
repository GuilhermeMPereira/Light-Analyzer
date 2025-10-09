
<?php
include('../connect.php');
include('../check.php');
extract($_POST);
$email = $_POST['email'];
$senha = md5($senha);
if(mysqli_query($con, "INSERT INTO `cliente` (`id`, `nome`, `telefone`, `email`,`senha`,
`criacao`) VALUES (NULL, '$firstname', '$telefone', '$email', '$senha',  NOW())")){
include_once('login.act.php');
header("location:../login.php");
}else{
    echo "Erro ao cadastrar";
    echo date("Y/m/d h:i:s");
}
?>
