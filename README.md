Proje 1
[22,27,16,2,18,6] -> Insertion Sort

ilk adımda sıralamadaki ilk sayı 22 dir. Amacımız ise sayılardan en küçüğünü 22 ile yer değiştirmek.
 [2,27,16,22,18,6]

ikinci adımda ise artık 2 sabit ve elimde 27 den küçük ve en küçük sayıyla 27nin yerini değiştiriyoruz.
 [2,6,16,22,18,27]

üçüncü adımda 16 en küçük sayıdır bu nedenle 4.sırayı incelemeye geçeriz 22 den küçük olan 18 ile 22 nin yeri değişir.
 [2,6,16,18,22,27]

son aşamada ise  dizinin tüm elemanları küçükten büyüğe tam olarak sıralanmış olduğundan işlem tamamlanmış olur.

Big-O gösterimini yazınız.

"n" elemanımız olduğunu varsaydığımızdan en küçük sayıyı bulmak için tüm elemanları kontrol ederiz ve sonrasında en küçük elemanı 
en başa yazarız. Artık sıradaki küçük sayıyı bulmak için en küçük verimiz hariç tüm verilere (n-1) bakarız. İşlem sonunda (n*(n+1)/2)den sonucumuzu 
O(n^2) buluruz.
