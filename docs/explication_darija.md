# شرح المشروع بالدارجة المغربية

## 1. Chno howa sujet ?

Sujet dyalna howa : **Détection de fraude dans les sinistres d'assurance par Machine Learning**.

يعني بغينا نديرو système ذكي كيساعد شركة التأمين تكتاشف الملفات اللي ممكن يكونو فيهم fraude.

## 2. Chno howa sinistre ?

Sinistre هو dossier ديال حادث ولا ضرر كيصرح به الزبون عند assurance.

مثلا :

- accident voiture ؛
- vehicle theft ؛
- collision ؛
- dégâts matériels ؛
- إصابات جسدية.

الزبون كيدير déclaration وكيطلب indemnisation، يعني التعويض.

## 3. Chno howa fraude ?

Fraude ف assurance هي ملي شي شخص يعطي معلومات غير صحيحة ولا يضخم المبلغ ولا يدير déclaration مشكوك فيها باش ياخذ تعويض ما مستحقوش.

مثلا :

- يصرح بحادث ما وقعش ؛
- يزيد ف montant dyal réclamation ؛
- يعطي معلومات ناقصة ولا كاذبة ؛
- يكون عندو سوابق مشبوهة.

## 4. 3lach assurance katحتاج détecter fraude ?

حيت fraude كتسبب خسائر مالية كبيرة لشركات التأمين. إلى كل dossier تراجع يدويا غادي ياخذ وقت بزاف.

Machine Learning كيعاون gestionnaire يعرف شنو هما dossiers اللي خاصهم contrôle الأول.

## 5. Kifach Machine Learning kay3awn ?

Machine Learning كيتعلم من data القديمة. عندنا dossiers معروفين واش هما fraude ولا non fraude.

Model كيشوف العلاقة بين المعلومات بحال :

- montant dyal claim ؛
- type dyal incident ؛
- severity dyal incident ؛
- nombre dyal véhicules ؛
- witnesses ؛
- police report ؛
- informations dyal assuré ؛
- informations dyal véhicule.

من بعد كيتعلم pattern ديال dossiers frauduleux.

## 6. Chno kaydir modèle ?

Model كيعطي لكل dossier واحد score سميتو `fraud_probability`.

مثلا :

- 0.15 يعني احتمال fraude ضعيف ؛
- 0.55 يعني dossier متوسط الخطورة ؛
- 0.85 يعني dossier مشكوك فيه بزاف.

## 7. Chno kat3ni fraud_probability ?

`fraud_probability` هي probabilité بين 0 و 1.

- قريبة من 0 : غالبا dossier عادي ؛
- قريبة من 1 : dossier فيه احتمال كبير ديال fraude.

## 8. Chno kat3ni risk_level ?

قسمنا dossiers لثلاثة niveaux :

- **Faible** : احتمال fraude أقل من 40%، traitement normal ؛
- **Moyen** : بين 40% و 70%، خاص vérification complémentaire ؛
- **Élevé** : أكثر من 70%، خاص contrôle approfondi par gestionnaire.

## 9. Chno kaydir dashboard ?

Dashboard هو interface تفاعلية ب Streamlit كيبين :

- شحال من dossier كاين ؛
- شحال من dossier مشكوك فيه ؛
- taux dyal fraude estimé ؛
- dossiers à risque élevé ؛
- graphes dyal risque ؛
- top 10 dossiers les plus suspects ؛
- tableau فيه dossiers و probability و action recommandée.

Gestionnaire يقدر يدير filtres ويشوف غير dossiers اللي بغا.

## 10. Kifach nشرح projet f soutenance ?

تقدر تقول :

"Projet dyali kayhder 3la détection de fraude f sinistres d'assurance. L'objectif howa n3awno compagnie d'assurance bach tكتاشف dossiers suspects rapidement. Khdit dataset فيه 1000 dossier، drt preprocessing, encodage, traitement des valeurs manquantes, men ba3d jربت plusieurs modèles Machine Learning. Model kayحسب probabilité de fraude لكل dossier، w kanclassiw dossier risk faible, moyen ou élevé. F lakhir drt dashboard Streamlit باش gestionnaire ychof résultats, filtres, KPIs, graphiques, top dossiers suspects w actions recommandées."

## 11. Conclusion simple

هاد المشروع ماشي كيحكم بوحدو على الزبون واش fraude ولا لا. هو غير outil d'aide à la décision كيساعد gestionnaire يركز على dossiers اللي فيهم risque كبير.
