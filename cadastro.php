<!DOCTYPE html>
<html lang="pt-br">
<head>
	<link rel="stylesheet" href="./assets/css/cad.css">
	<link rel="icon" href="./assets/img/Camera_Icon-removebg-preview.png">
	<title>Ender Softwares - Cadastro</title>
</head>
<body>
	<section>
		<div class="form-box">
			<div class="form-value">
				<div class="logo">
				<a href="index.html"> <img src="./assets/img/Camera_Icon-removebg-preview.png" ></a>
				</div>
				<form action="php/cad.act.php" method="POST" >
					
					<input type="hidden" name="tipo" value="programador">
					<div class="inputbox">
						<input id="firstname" type="text" name="firstname" required>
						<label for="firstname">Nome</label>
					</div>

					<div class="inputbox">
						<input  id="telefone" type="text" name="telefone" required>
						<label for="telefone">Telefone</label>
					</div>

					<div class="inputbox">  
						<input id="email" type="email" name="email" required>
                    <label for="email">Email</label>
                </div>


					<div class="inputbox">
						<input id="senha" type="password" name="senha" required>
						<label>Senha</label>
					</div>



					<div class="forget">
						<label for="">Ja tem conta? <a href="login.php">Entrar</a></label>

					</div>
					
					<button type="submit" class="botao">Continuar</a></button>

				</form>
			</div>
		</div>
	</section>
	<script type="module" src="https://unpkg.com/ionicons@5.5.2/dist/ionicons/ionicons.esm.js"></script>
	<script nomodule src="https://unpkg.com/ionicons@5.5.2/dist/ionicons/ionicons.js"></script>
</body>
</html>