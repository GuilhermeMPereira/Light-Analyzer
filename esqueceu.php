<!DOCTYPE html>
<html lang="pt-br">

<!-- <?php
include("connect.php");
include("check.php");

if(isset($_POST['ok'])){


    if(!filter_var($email, FILTER_VALIDATE_EMAIL)){
        $erro [] = "E-mail inválido.";
    }

    $sql_code = "SELECT senha, id_cliente FROM user WHERE email = '$_SESSION[email]'";
    $sql_query = $mysqli->query($sql_code) or die($mysqli->error);
    $dado = $sql_query->fetch_assoc();
    $total = $sql_query->num_rows;

    if($total == 0){
        $erro [] = "E-mail não existe.";
    }


    if(count($erro) == 0 && $total > 0){

    $novasenha = substr(md5(time()), 0, 6);
    $nscriptografada = md5(md5($novasenha));
    $email = $mysqli->escape_string($_POST['email']);

if(1 == 1){

    $sql_code = "UPDATE user SET senha = '$nscriptografada' WHERE email = '$email'";
    $sql_query = $mysqli->query($sql_code) or die ($mysqli->error);

    if($sql_query){
        $erro[] = "Senha Alterada com Sucesso";
    }
        }
    }
}
?> -->

<head>
	<link rel="stylesheet" href="./register/cadastroFisica/css/cadf.css">
	<link rel="icon" href="../../assets/img/icon.png">
	<title>Ender Softwares - Cadastro Pessoa Física</title>
</head>

<body>

	<section>
		<div class="form-box">
			<div class="form-value">
				<div class="logo">
				<a href="index.php"> <img src="./assets/img/logo.png" alt="Ender Softwares"> </a>
				</div>
				<form action="esqueceu.act.php" method="POST">
					<h2>Trocar a Senha</h2>
					<input type="hidden" name="tipo" value="cliente">
					
					<div class="inputbox">  
						<input id="email" type="email" name="email" required>
                    <label for="email">Email</label>
                </div>

					<a href="index.php"><button type="submit" name="ok"class="botao">Continuar</button></a>

				</form>
			</div>
		</div>
	</section>
	<script type="module" src="https://unpkg.com/ionicons@5.5.2/dist/ionicons/ionicons.esm.js"></script>
	<script nomodule src="https://unpkg.com/ionicons@5.5.2/dist/ionicons/ionicons.js"></script>
</body>
</html>