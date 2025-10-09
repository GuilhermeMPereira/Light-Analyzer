<?php
    include('connect.php');
    if (isset($_COOKIE["ID"])){
        $user_id = $_COOKIE["ID"];

        $stmt = $con->prepare("UPDATE user SET online = now() WHERE id_cliente = ?");
        $stmt->bind_param("i", $user_id);
        $stmt->execute();

        $busca = "SELECT * FROM `user`
        LEFT JOIN `endereco`
        ON user.id_cliente = user.id_cliente
        LEFT JOIN `habilidades`
        ON endereco.id_cliente = habilidades.id_cliente
        WHERE user.id_cliente = '$user_id'";

        $resultado = mysqli_query($con, $busca);
        $user = mysqli_fetch_array($resultado);
    }
?>